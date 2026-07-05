#!/usr/bin/env python3
"""Turn aligned per-episode tables into stacked (X, Y) training samples.

    build_dataset.py [ALIGNED_NPZ ...]   # default: all data/aligned/**/*.aligned.npz

Per aligned episode it emits one dataset file:
    X [N,208]  = concat[f_{t-3s}, f_{t-2s}, f_{t-1s}, f_t]  (frame t is the LAST 52)
                 each frame f = [act_pos13 | act_err13 | act_vel13 | action13]
    Y [N,16]   = gt_pos at (t + target_horizon)
    t [N]      = grid time of the current frame
Stacking never crosses a segment boundary (a data gap) or an episode boundary.
Writes data/dataset/<session>/<episode>.h5 (or .npz) + data/dataset/manifest.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

try:
    import h5py
    _HAVE_H5 = True
except Exception:  # noqa: BLE001
    _HAVE_H5 = False


def build_one(aligned_path: pathlib.Path, pipeline: dict, joints: dict | None = None) -> dict:
    joints = joints or cl.load_joints()
    d = np.load(aligned_path, allow_pickle=True)
    L = int(pipeline["stack_len"])
    stride = int(pipeline["stack_stride"])
    horizon = int(pipeline["target_horizon"])

    # --- Policy-frame construction (mirrors get_proprioception in the policy node) ------
    # Aligned columns are raw radians in actuator_order; reorder to policy_joint_order,
    # then normalise pos/vel/action by limits (err stays raw radians). J0 is left as the
    # summed value (no x2) — the network learns the coupling into 16 individual joints.
    perm = cl.policy_perm(joints)
    lower, upper, vel = cl.policy_limits(joints)
    pos = d["act_pos"][:, perm]                    # process_value (J0 summed)
    velv = d["act_vel"][:, perm]                   # process_value_dot
    err = d["act_err"][:, perm]                    # controller error, radians
    act = d["action"][:, perm]                     # set_point (J0 summed)
    pos_n = cl.normalise(pos, lower, upper)
    vel_n = cl.normalise(velv, -vel, vel)
    act_n = cl.normalise(act, lower, upper)
    # group order MUST match cl.FRAME_FEATURE_GROUPS: pos, vel, err, action
    F = np.hstack([pos_n, vel_n, err, act_n])
    assert F.shape[1] == cl.FRAME_DIM, F.shape
    seg = d["seg_id"]
    gt = d["gt_pos"]
    t = d["t"]
    T = F.shape[0]

    span = (L - 1) * stride
    X, Y, tc, tt = [], [], [], []
    for cur in range(span, T - horizon):
        idxs = [cur - k * stride for k in range(L - 1, -1, -1)]  # oldest..current
        tgt = cur + horizon
        s = seg[cur]
        if s < 0:
            continue
        if any(seg[i] != s for i in idxs) or seg[tgt] != s:
            continue
        X.append(np.concatenate([F[i] for i in idxs]))
        Y.append(gt[tgt])
        tc.append(t[cur])
        tt.append(t[tgt])

    X = np.asarray(X, np.float32).reshape(-1, cl.FRAME_DIM * L)
    Y = np.asarray(Y, np.float32).reshape(-1, cl.N_JOINTS)
    return {
        "X": X, "Y": Y,
        "t": np.asarray(tc, np.float64), "t_target": np.asarray(tt, np.float64),
    }


def _write(path: pathlib.Path, arrays: dict, attrs: dict, fmt: str) -> pathlib.Path:
    if fmt == "hdf5" and _HAVE_H5:
        p = path.with_suffix(".h5")
        with h5py.File(p, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=v, compression="gzip")
            for k, v in attrs.items():
                f.attrs[k] = v
        return p
    p = path.with_suffix(".npz")
    np.savez_compressed(p, **arrays, **{f"attr_{k}": v for k, v in attrs.items()})
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aligned", nargs="*", help="*.aligned.npz (default: all under data/aligned)")
    args = ap.parse_args()

    pipeline = cl.load_pipeline()
    joints = cl.load_joints()
    fmt = pipeline.get("output_format", "hdf5")
    if fmt == "hdf5" and not _HAVE_H5:
        print("WARNING: h5py not installed -> falling back to .npz output", file=sys.stderr)

    paths = [pathlib.Path(p) for p in args.aligned] or sorted(
        pathlib.Path(cl.REPO_ROOT / "data" / "aligned").glob("**/*.aligned.npz"))
    if not paths:
        print("no aligned files found; run align.py first", file=sys.stderr)
        return 1

    manifest = {
        "input_dim": cl.FRAME_DIM * int(pipeline["stack_len"]),
        "output_dim": cl.N_JOINTS,
        "stack_len": int(pipeline["stack_len"]),
        "stack_stride": int(pipeline["stack_stride"]),
        "target_horizon": int(pipeline["target_horizon"]),
        "dataset_rate": float(pipeline["dataset_rate"]),
        "frame_layout": "[pos_norm|vel_norm|err|action] x " + str(cl.FRAME_FEATURE_GROUPS),
        "actuator_order": joints["actuator_order"],
        "policy_joint_order": joints["policy_joint_order"],
        "joint_order": joints["joint_order"],
        "input_columns": cl.input_column_names(joints, int(pipeline["stack_len"])),
        "episodes": [],
    }

    for ap_path in paths:
        arrays = build_one(ap_path, pipeline, joints)
        session = ap_path.parent.name
        episode = ap_path.name.replace(".aligned.npz", "")
        outdir = cl.REPO_ROOT / "data" / "dataset" / session
        outdir.mkdir(parents=True, exist_ok=True)
        attrs = {"input_dim": manifest["input_dim"], "output_dim": manifest["output_dim"],
                 "session": session, "episode": episode}
        out = _write(outdir / episode, arrays, attrs, fmt)
        manifest["episodes"].append({
            "session": session, "episode": episode,
            "file": str(out.relative_to(cl.REPO_ROOT)),
            "n_samples": int(arrays["X"].shape[0]),
            "aligned": str(ap_path.relative_to(cl.REPO_ROOT)),
        })
        print(f"{out}  X{arrays['X'].shape} Y{arrays['Y'].shape}")

    man_path = cl.REPO_ROOT / "data" / "dataset" / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(e["n_samples"] for e in manifest["episodes"])
    print(f"\nwrote {man_path}  ({len(manifest['episodes'])} episodes, {total} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
