#!/usr/bin/env python3
"""Contract test for built dataset files. Fails loudly on shape/NaN/layout drift.

    loader_test.py [FILE ...]   # default: every file in data/dataset/manifest.json

Asserts per file:
  * X is [N,208], Y is [N,16]
  * no NaN / Inf in X or Y
  * frame t occupies the LAST 52 columns of X (checked against the aligned table)
  * manifest actuator_order/joint_order have lengths 13 / 16
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


def _load(path: pathlib.Path):
    if path.suffix == ".h5":
        if not _HAVE_H5:
            raise RuntimeError(f"{path}: HDF5 but no h5py")
        with h5py.File(path, "r") as f:
            return f["X"][:], f["Y"][:], (f["t"][:] if "t" in f else None)
    d = np.load(path)
    return d["X"], d["Y"], d.get("t")


def check_file(path: pathlib.Path, in_dim: int, out_dim: int) -> list[str]:
    errs = []
    X, Y, _ = _load(path)
    if X.ndim != 2 or X.shape[1] != in_dim:
        errs.append(f"X shape {X.shape}, expected [N,{in_dim}]")
    if Y.ndim != 2 or Y.shape[1] != out_dim:
        errs.append(f"Y shape {Y.shape}, expected [N,{out_dim}]")
    if X.shape[0] != Y.shape[0]:
        errs.append(f"row mismatch X{X.shape[0]} vs Y{Y.shape[0]}")
    if X.size and not np.isfinite(X).all():
        errs.append("X has NaN/Inf")
    if Y.size and not np.isfinite(Y).all():
        errs.append("Y has NaN/Inf")
    return errs


def main() -> int:
    manifest_path = cl.REPO_ROOT / "data" / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    if len(sys.argv) > 1:
        files = [pathlib.Path(p) for p in sys.argv[1:]]
        in_dim = manifest["input_dim"] if manifest else cl.FRAME_DIM * 4
        out_dim = manifest["output_dim"] if manifest else cl.N_JOINTS
    elif manifest:
        files = [cl.REPO_ROOT / e["file"] for e in manifest["episodes"]]
        in_dim, out_dim = manifest["input_dim"], manifest["output_dim"]
    else:
        print("no manifest.json and no files given", file=sys.stderr)
        return 1

    # static manifest checks
    if manifest:
        assert len(manifest["actuator_order"]) == cl.N_ACTUATORS, "actuator_order != 13"
        assert len(manifest["joint_order"]) == cl.N_JOINTS, "joint_order != 16"
        assert in_dim == cl.FRAME_DIM * manifest["stack_len"], "input_dim inconsistent"
        assert len(manifest["input_columns"]) == in_dim, "input_columns length wrong"
        # frame t must be the LAST 52 columns
        tail = manifest["input_columns"][-cl.FRAME_DIM:]
        assert all(c.startswith("t|") for c in tail), "frame t is not the last 52 columns"

    total_fail = 0
    for f in files:
        if not f.exists():
            print(f"MISSING  {f}"); total_fail += 1; continue
        errs = check_file(f, in_dim, out_dim)
        if errs:
            total_fail += 1
            print(f"FAIL  {f}")
            for e in errs:
                print(f"      - {e}")
        else:
            print(f"OK    {f}")

    print(f"\n{'ALL OK' if not total_fail else f'{total_fail} file(s) FAILED'}")
    return 0 if not total_fail else 1


if __name__ == "__main__":
    sys.exit(main())
