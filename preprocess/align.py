#!/usr/bin/env python3
"""Resample every stream from one bag onto a single uniform grid at dataset_rate.

    align.py BAG [--out DIR]        # writes <episode>.aligned.npz, merges QC into
                                     # meta/<session>/<episode>.json (the same sidecar
                                     # record_episode.sh/run_episode.py already wrote)

Aligned table columns (all on the same grid, actuator/joint canonical order):
    t         [T]
    act_pos   [T,13]  = process_value        (J0 = summed value straight from driver)
    act_err   [T,13]  = error
    act_vel   [T,13]  = process_value_dot
    action    [T,13]  = set_point            (commanded target)
    gt_pos    [T,16]  = /joint_states.position   (the 16 outputs)
    gt_vel    [T,16]  = /joint_states.velocity    (context)
    gt_effort [T,16]  = /joint_states.effort      (context)
    command   [T,13]  = controller .command / PWM (context)
    valid     [T]     bool: False where any stream had no sample within max_gap_ms
    seg_id    [T]     int : contiguous valid-run id (-1 where invalid)

Segments (seg_id) are how build_dataset.py avoids stacking across a data gap.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
from preprocess.parse_bag import parse_bag  # noqa: E402


def _resample(grid: np.ndarray, src_t: np.ndarray, src_v: np.ndarray,
              mode: str) -> np.ndarray:
    """Interpolate src_v(src_t) onto grid. mode: 'linear' | 'zoh' | 'nearest'."""
    if src_t.size == 0:
        return np.full(grid.shape, np.nan)
    order = np.argsort(src_t, kind="stable")
    st, sv = src_t[order], src_v[order]
    if mode == "linear":
        return np.interp(grid, st, sv)
    # zoh: last sample with st <= g ; nearest: closest sample
    idx = np.searchsorted(st, grid, side="right") - 1
    idx = np.clip(idx, 0, st.size - 1)
    if mode == "zoh":
        return sv[idx]
    if mode == "nearest":
        nxt = np.clip(idx + 1, 0, st.size - 1)
        take_next = np.abs(st[nxt] - grid) < np.abs(grid - st[idx])
        idx = np.where(take_next, nxt, idx)
        return sv[idx]
    raise ValueError(f"unknown interp mode {mode!r}")


def _nn_distance(grid: np.ndarray, src_t: np.ndarray) -> np.ndarray:
    """Distance (s) from each grid point to the nearest source stamp."""
    if src_t.size == 0:
        return np.full(grid.shape, np.inf)
    st = np.sort(src_t)
    idx = np.clip(np.searchsorted(st, grid), 0, st.size - 1)
    prev = np.clip(idx - 1, 0, st.size - 1)
    return np.minimum(np.abs(st[idx] - grid), np.abs(grid - st[prev]))


def align_parsed(parsed: dict, pipeline: dict, joints: dict) -> dict:
    acts = joints["actuator_order"]
    interp = pipeline["interp"]
    rate = float(pipeline["dataset_rate"])
    max_gap = float(pipeline["max_gap_ms"]) / 1e3

    js = parsed["joint_states"]
    # Common time span across all present streams.
    starts = [js["t"][0]]
    ends = [js["t"][-1]]
    for a in acts:
        t = parsed["controller"][a]["t"]
        if t.size:
            starts.append(t[0]); ends.append(t[-1])
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        raise RuntimeError("no overlapping time span across streams")
    grid = np.arange(t0, t1, 1.0 / rate)

    T = grid.size
    act_pos = np.empty((T, len(acts)))
    act_err = np.empty((T, len(acts)))
    act_vel = np.empty((T, len(acts)))
    action = np.empty((T, len(acts)))
    command = np.empty((T, len(acts)))
    nn = _nn_distance(grid, js["t"])  # start gap tracking with joint_states

    for i, a in enumerate(acts):
        c = parsed["controller"][a]
        act_pos[:, i] = _resample(grid, c["t"], c["process_value"], interp["process_value"])
        act_err[:, i] = _resample(grid, c["t"], c["error"], interp["error"])
        act_vel[:, i] = _resample(grid, c["t"], c["process_value_dot"],
                                  interp["process_value_dot"])
        action[:, i] = _resample(grid, c["t"], c["set_point"], interp["set_point"])
        command[:, i] = _resample(grid, c["t"], c["command"], interp["command"])
        nn = np.maximum(nn, _nn_distance(grid, c["t"]))

    gt_pos = np.column_stack([
        _resample(grid, js["t"], js["position"][:, j], interp["joint_position"])
        for j in range(cl.N_JOINTS)])
    gt_vel = np.column_stack([
        _resample(grid, js["t"], js["velocity"][:, j], interp["joint_velocity"])
        for j in range(cl.N_JOINTS)])
    gt_effort = np.column_stack([
        _resample(grid, js["t"], js["effort"][:, j], interp["joint_effort"])
        for j in range(cl.N_JOINTS)])

    valid = np.isfinite(nn) & (nn <= max_gap)
    # Label each contiguous run of valid grid points with a consecutive seg_id (>=0);
    # invalid points get -1. A rising edge (invalid->valid) starts a new segment.
    seg_id = np.full(T, -1, int)
    sid = -1
    for k in range(T):
        if valid[k]:
            if k == 0 or not valid[k - 1]:
                sid += 1
            seg_id[k] = sid

    return {
        "t": grid,
        "act_pos": act_pos, "act_err": act_err, "act_vel": act_vel, "action": action,
        "gt_pos": gt_pos, "gt_vel": gt_vel, "gt_effort": gt_effort, "command": command,
        "valid": valid, "seg_id": seg_id,
        "actuator_order": np.array(acts),
        "joint_order": np.array(joints["joint_order"]),
        "dataset_rate": rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out", default=None,
                    help="output dir (default data/aligned/<session>/)")
    args = ap.parse_args()

    pipeline = cl.load_pipeline()
    joints = cl.load_joints()
    bag = pathlib.Path(args.bag).resolve()

    parsed = parse_bag(bag)
    aligned = align_parsed(parsed, pipeline, joints)

    session = bag.parent.name
    outdir = pathlib.Path(args.out) if args.out else cl.REPO_ROOT / "data" / "aligned" / session
    outdir.mkdir(parents=True, exist_ok=True)
    stem = bag.stem
    npz = outdir / f"{stem}.aligned.npz"
    np.savez_compressed(npz, **{k: v for k, v in aligned.items()})

    n_valid = int(aligned["valid"].sum())
    n_seg = int(aligned["seg_id"].max()) + 1 if (aligned["seg_id"] >= 0).any() else 0
    aligned_info = {
        "dataset_rate": aligned["dataset_rate"],
        "n_grid": int(aligned["t"].size), "n_valid": n_valid, "n_segments": n_seg,
        "parse_report": parsed["report"],
    }

    # Merge into the SAME meta/<session>/<stem>.json that record_episode.sh/
    # run_episode.py wrote at collection time, rather than writing a separate sidecar --
    # one JSON per episode ends up describing everything about it (config + QC).
    meta_dir = cl.REPO_ROOT / "meta" / session
    meta_path = meta_dir / f"{stem}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta = {"bag": str(bag)}
        print(f"NOTE: no existing {meta_path} (pre-meta/-layout session?) -- creating "
              f"one with align-only fields.", file=sys.stderr)
    meta["aligned_npz"] = str(npz)
    meta["aligned"] = aligned_info
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    print(f"wrote {npz}  (grid={aligned['t'].size}, valid={n_valid}, segments={n_seg})")
    print(f"updated {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
