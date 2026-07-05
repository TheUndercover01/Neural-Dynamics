#!/usr/bin/env python3
"""Split episodes and compute a normalization scaler from TRAIN episodes only.

    normalize.py     # reads data/dataset/manifest.json, writes scaler.json

Split is BY EPISODE (never by frame) to prevent temporal leakage: the same 60 s episode
never contributes rows to two splits. Scaler mean/std are over the train split's X and Y.
"""
from __future__ import annotations

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


def _load_xy(path: pathlib.Path):
    if path.suffix == ".h5":
        if not _HAVE_H5:
            raise RuntimeError(f"{path} is HDF5 but h5py is unavailable")
        with h5py.File(path, "r") as f:
            return f["X"][:], f["Y"][:]
    d = np.load(path)
    return d["X"], d["Y"]


def assign_splits(episodes: list, split: dict) -> dict:
    keys = [f"{e['session']}/{e['episode']}" for e in episodes]
    rng = np.random.default_rng(int(split.get("seed", 0)))
    order = rng.permutation(len(keys))
    n = len(keys)
    n_tr = int(round(split["train"] * n))
    n_va = int(round(split["val"] * n))
    assign = {}
    for rank, idx in enumerate(order):
        if rank < n_tr:
            assign[keys[idx]] = "train"
        elif rank < n_tr + n_va:
            assign[keys[idx]] = "val"
        else:
            assign[keys[idx]] = "test"
    return assign


def main() -> int:
    pipeline = cl.load_pipeline()
    man_path = cl.REPO_ROOT / "data" / "dataset" / "manifest.json"
    if not man_path.exists():
        print("no manifest.json; run build_dataset.py first", file=sys.stderr)
        return 1
    manifest = json.loads(man_path.read_text())
    episodes = manifest["episodes"]
    if not episodes:
        print("manifest has no episodes", file=sys.stderr)
        return 1

    assign = assign_splits(episodes, pipeline["split"])

    # Streaming mean/std over train rows only (Welford-ish via sums to keep it simple).
    xs = ys = None
    x_sum = x_sq = y_sum = y_sq = None
    n_rows = 0
    for e in episodes:
        key = f"{e['session']}/{e['episode']}"
        if assign[key] != pipeline.get("normalize_on", "train"):
            continue
        X, Y = _load_xy(cl.REPO_ROOT / e["file"])
        if X.size == 0:
            continue
        X = X.astype(np.float64); Y = Y.astype(np.float64)
        if x_sum is None:
            x_sum = X.sum(0); x_sq = (X ** 2).sum(0)
            y_sum = Y.sum(0); y_sq = (Y ** 2).sum(0)
        else:
            x_sum += X.sum(0); x_sq += (X ** 2).sum(0)
            y_sum += Y.sum(0); y_sq += (Y ** 2).sum(0)
        n_rows += X.shape[0]

    if not n_rows:
        print("no rows in normalize split; check split ratios / sample counts", file=sys.stderr)
        return 1

    x_mean = x_sum / n_rows
    x_std = np.sqrt(np.maximum(x_sq / n_rows - x_mean ** 2, 0.0))
    y_mean = y_sum / n_rows
    y_std = np.sqrt(np.maximum(y_sq / n_rows - y_mean ** 2, 0.0))
    eps = 1e-6
    x_std = np.where(x_std < eps, 1.0, x_std)   # guard constant features
    y_std = np.where(y_std < eps, 1.0, y_std)

    scaler = {
        "normalize_on": pipeline.get("normalize_on", "train"),
        "n_rows": int(n_rows),
        "input_dim": int(x_mean.size), "output_dim": int(y_mean.size),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "split": {k: assign[k] for k in sorted(assign)},
    }
    out = cl.REPO_ROOT / "data" / "dataset" / "scaler.json"
    out.write_text(json.dumps(scaler, indent=2))
    counts = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
    print(f"wrote {out}  (episodes: {counts}, normalize rows={n_rows})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
