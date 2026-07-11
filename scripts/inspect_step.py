#!/usr/bin/env python3
"""Trace ONE step command through every pipeline stage for the same joint, so you can
confirm the data is consistent from raw bag -> aligned grid -> built (X,Y) dataset.

Read-only: runs no pipeline stage itself, just re-reads what parse_bag/align/build_dataset
already produced (or, for the raw stage, re-parses the original bag on the fly).

    scripts/inspect_step.py --aligned data/aligned/<session>/<ep>.aligned.npz --joint rh_FFJ3
    scripts/inspect_step.py --aligned data/aligned/<session>/<ep>.aligned.npz --joint rh_FFJ2
                                                                        # (coupled -> rh_FFJ0)

For a coupled actuator (rh_FFJ0/rh_MFJ0/rh_RFJ0), pass either the actuator name or either
of its driven joint names (rh_FFJ1/rh_FFJ2 etc.) — both resolve to the same command trace.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
from diagnostics.latency_timeconstant import detect_steps  # noqa: E402

try:
    import h5py
    _HAVE_H5 = True
except Exception:  # noqa: BLE001
    _HAVE_H5 = False


def resolve_actuator(joint_arg: str, joints: dict) -> tuple[str, list[str]]:
    """joint_arg -> (actuator_name, [driven joint name(s)])."""
    acts = list(joints["actuator_order"])
    coupling = cl.coupled_actuators(joints)
    if joint_arg in acts:
        driven = list(coupling.get(joint_arg, [joint_arg]))
        return joint_arg, driven
    for a, driven in coupling.items():
        if joint_arg in driven:
            return a, list(driven)
    if joint_arg in joints["joint_order"]:
        return joint_arg, [joint_arg]
    raise SystemExit(f"--joint {joint_arg!r} not found in actuator_order/joint_order/coupling")


def find_bag_path(aligned_path: pathlib.Path, cl_root: pathlib.Path) -> pathlib.Path | None:
    """Looks up meta/<session>/<stem>.json (the JSON sidecar record_episode.sh/
    run_episode.py wrote, later merged with align.py's QC) -- NOT next to the aligned
    npz itself, see README.md's Layout section for why raw/aligned dirs hold only
    binaries."""
    session = aligned_path.parent.name
    stem = aligned_path.name
    if stem.endswith(".aligned.npz"):
        stem = stem[: -len(".aligned.npz")]
    sidecar = cl_root / "meta" / session / f"{stem}.json"
    if not sidecar.exists():
        return None
    try:
        bag_field = pathlib.Path(json.loads(sidecar.read_text())["bag"])
        # "bag" is stored as a basename (relative to data/raw/<session>/), not absolute.
        bag = bag_field if bag_field.is_absolute() else cl_root / "data" / "raw" / session / bag_field
        return bag if bag.exists() else None
    except Exception:  # noqa: BLE001
        return None


def find_dataset_file(aligned_path: pathlib.Path, cl_root: pathlib.Path) -> pathlib.Path | None:
    man_path = cl_root / "data" / "dataset" / "manifest.json"
    if not man_path.exists():
        return None
    manifest = json.loads(man_path.read_text())
    aligned_rel = str(aligned_path.resolve().relative_to(cl_root)) if aligned_path.is_absolute() \
        else str(aligned_path)
    for e in manifest["episodes"]:
        if e["aligned"] == aligned_rel or e["aligned"].endswith(aligned_path.name):
            return cl_root / e["file"]
    return None


def load_xy(path: pathlib.Path):
    if path.suffix == ".h5":
        if not _HAVE_H5:
            raise RuntimeError(f"{path} is HDF5 but h5py is unavailable")
        with h5py.File(path, "r") as f:
            return f["X"][:], f["Y"][:], f["t"][:]
    d = np.load(path)
    return d["X"], d["Y"], d["t"]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(c.rjust(w) for c, w in zip(r, widths)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", required=True, help="path to *.aligned.npz")
    ap.add_argument("--joint", required=True, help="actuator or joint name, e.g. rh_FFJ3")
    ap.add_argument("--window", type=int, default=15,
                    help="frames to show before/after the step at each stage (default 15)")
    ap.add_argument("--step-thresh", type=float, default=None,
                    help="override config/latency.yaml's step_thresh for detection")
    args = ap.parse_args()

    joints = cl.load_joints()
    latency_cfg = cl.load_latency()
    step_thresh = args.step_thresh if args.step_thresh is not None else latency_cfg["step_thresh"]

    actuator, driven = resolve_actuator(args.joint, joints)
    acts = list(joints["actuator_order"])
    jorder = list(joints["joint_order"])
    ai = acts.index(actuator)
    ji = {j: jorder.index(j) for j in driven}

    aligned_path = pathlib.Path(args.aligned).resolve()
    d = np.load(aligned_path, allow_pickle=True)
    rate = float(d["dataset_rate"])
    seg_id = d["seg_id"]

    steps = detect_steps(d["action"][:, ai], seg_id, step_thresh, latency_cfg["hold_frames"])
    if not steps:
        print(f"No clean step detected for {actuator} in {aligned_path.name} "
              f"(step_thresh={step_thresh} rad, hold_frames={latency_cfg['hold_frames']}).")
        return 2
    step = steps[0]
    sf = step["step_frame"]
    print(f"actuator={actuator}  driven joint(s)={driven}  "
          f"step_frame={sf}  t={d['t'][sf]:.4f}s  magnitude={step['magnitude']:+.4f} rad  "
          f"(aligned @ {rate:g} Hz)\n")

    w = args.window
    lo, hi = max(0, sf - w), min(d["t"].size, sf + w)

    # --- Stage 1: raw bag (re-parsed) ------------------------------------------------------
    print("=" * 88)
    print("STAGE 1 — raw bag (preprocess/parse_bag.py)")
    print("=" * 88)
    bag_path = find_bag_path(aligned_path, cl.REPO_ROOT)
    if bag_path is None:
        print("  (no meta/<session>/<episode>.json sidecar / bag found — skipping)")
    else:
        from preprocess.parse_bag import parse_bag
        parsed = parse_bag(bag_path)
        c = parsed["controller"][actuator]
        t_event = float(d["t"][sf])
        mask = (c["t"] >= t_event - w / rate) & (c["t"] <= t_event + w / rate)
        idxs = np.where(mask)[0]
        print(f"  bag: {bag_path}")
        print(f"  {c['t'].size} raw samples for {actuator}; {idxs.size} within +/-{w} "
              f"aligned-frames of the step\n")
        rows = [[f"{c['t'][i]:.4f}", f"{c['set_point'][i]:+.4f}", f"{c['process_value'][i]:+.4f}"]
                for i in idxs]
        print_table(["t (s)", "set_point", "process_value"], rows)

    # --- Stage 2: aligned grid (preprocess/align.py) ---------------------------------------
    print("\n" + "=" * 88)
    print("STAGE 2 — aligned grid (preprocess/align.py), 60Hz-resampled")
    print("=" * 88)
    headers = ["frame", "t (s)", "action(ZOH)", "act_pos(lin)"] + [f"gt_pos:{j}" for j in driven]
    rows = []
    for i in range(lo, hi):
        row = [str(i), f"{d['t'][i]:.4f}", f"{d['action'][i, ai]:+.4f}", f"{d['act_pos'][i, ai]:+.4f}"]
        row += [f"{d['gt_pos'][i, ji[j]]:+.4f}" for j in driven]
        marker = "  <- STEP" if i == sf else ""
        rows.append(row)
        if marker:
            rows[-1][0] += marker
    print_table(headers, rows)
    if actuator in cl.coupled_actuators(joints):
        summed = sum(d["gt_pos"][sf, ji[j]] for j in driven)
        print(f"\n  coupling check @ step frame: act_pos({actuator})={d['act_pos'][sf, ai]:+.4f}  "
              f"vs  sum(gt_pos{driven})={summed:+.4f}  "
              f"|diff|={abs(d['act_pos'][sf, ai] - summed):.4f}")

    # --- Stage 3: built dataset (preprocess/build_dataset.py) ------------------------------
    print("\n" + "=" * 88)
    print("STAGE 3 — built dataset (preprocess/build_dataset.py)")
    print("=" * 88)
    ds_path = find_dataset_file(aligned_path, cl.REPO_ROOT)
    if ds_path is None or not ds_path.exists():
        print("  (no matching entry in data/dataset/manifest.json — run build_dataset.py first)")
    else:
        X, Y, t = load_xy(ds_path)
        pipeline = cl.load_pipeline()
        L = int(pipeline["stack_len"])
        cols = cl.frame_column_names(joints)  # 52 names, this stage's per-frame layout
        # policy-frame slot this actuator's `action` feature lives in (frame t = last 52 cols)
        perm = cl.policy_perm(joints)  # actuator_order index -> policy slot, per policy col
        policy_slot = perm.index(ai)
        action_col = cols.index(f"action:{joints['policy_joint_order'][policy_slot]}")
        row_idx = int(np.argmin(np.abs(t - d["t"][sf])))
        print(f"  dataset file: {ds_path}")
        print(f"  X{X.shape} Y{Y.shape}  nearest row to step = {row_idx}  "
              f"(row t={t[row_idx]:.4f}s vs aligned step t={d['t'][sf]:.4f}s)")
        rows = []
        for r in range(max(0, row_idx - 3), min(X.shape[0], row_idx + 4)):
            frame_t_action = X[r, -cl.FRAME_DIM + action_col]  # frame t = last 52 cols
            denorm = 0.5 * (frame_t_action + 1.0) * (
                cl.policy_limits(joints)[1][policy_slot] - cl.policy_limits(joints)[0][policy_slot]
            ) + cl.policy_limits(joints)[0][policy_slot]
            gt_next = [f"{Y[r, jorder.index(j)]:+.4f}" for j in driven]
            marker = "  <- nearest to step" if r == row_idx else ""
            rows.append([str(r), f"{t[r]:.4f}", f"{frame_t_action:+.4f}", f"{denorm:+.4f}"] + gt_next + [marker])
        print_table(["row", "t(s)", "action_norm[-1,1]", "action_denorm(rad)"] +
                   [f"Y:{j}(t+{pipeline['target_horizon']})" for j in driven] + [""], rows)
        print(f"\n  consistency check: action_denorm should match aligned action(rad) above "
              f"(within normalize round-trip float error).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
