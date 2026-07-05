#!/usr/bin/env python3
"""Per-episode QC report from an aligned table.

    report.py ALIGNED_NPZ [--out DIR]

Produces (under data/dataset/qc/<episode>/):
  * setpoint_vs_process_<ACT>.png  — commanded vs measured per actuator (the tendon gap)
  * coupling_<ACT>.png             — J0 process_value vs (J1+J2) from joint_states
  * error_hist.png                 — per-actuator error distribution
  * coverage.png                   — joint-space coverage per actuator
  * index.html                     — tables (rates, gaps) + all figures inline

Falls back to an HTML-with-tables-only report if matplotlib is unavailable.
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False


def _b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aligned")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    joints = cl.load_joints()
    acts = list(joints["actuator_order"])
    coupling = cl.coupled_actuators(joints)
    jorder = list(joints["joint_order"])

    ap_path = pathlib.Path(args.aligned).resolve()
    d = np.load(ap_path, allow_pickle=True)
    t = d["t"] - d["t"][0]
    outdir = pathlib.Path(args.out) if args.out else (
        cl.REPO_ROOT / "data" / "dataset" / "qc" / ap_path.name.replace(".aligned.npz", ""))
    outdir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, pathlib.Path]] = []

    if _HAVE_MPL:
        # set_point vs process_value grid
        ncol = 4
        nrow = int(np.ceil(len(acts) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False)
        for i, a in enumerate(acts):
            ax = axs[i // ncol][i % ncol]
            ax.plot(t, d["action"][:, i], lw=0.8, label="set_point")
            ax.plot(t, d["act_pos"][:, i], lw=0.8, label="process_value")
            ax.set_title(a, fontsize=8)
            ax.tick_params(labelsize=6)
        axs[0][0].legend(fontsize=6)
        for j in range(len(acts), nrow * ncol):
            axs[j // ncol][j % ncol].axis("off")
        fig.tight_layout()
        p = outdir / "setpoint_vs_process.png"; fig.savefig(p, dpi=90); plt.close(fig)
        figs.append(("set_point vs process_value (tendon gap)", p))

        # coupling overlays
        jidx = {n: k for k, n in enumerate(jorder)}
        fig, axs = plt.subplots(1, len(coupling), figsize=(4 * len(coupling), 2.6),
                                squeeze=False)
        for k, (a, (j1, j2)) in enumerate(coupling.items()):
            ax = axs[0][k]
            ai = acts.index(a)
            summed = d["gt_pos"][:, jidx[j1]] + d["gt_pos"][:, jidx[j2]]
            ax.plot(t, d["act_pos"][:, ai], lw=0.8, label="process_value")
            ax.plot(t, summed, lw=0.8, ls="--", label=f"{j1}+{j2}")
            ax.set_title(a, fontsize=8); ax.tick_params(labelsize=6); ax.legend(fontsize=6)
        fig.tight_layout()
        p = outdir / "coupling.png"; fig.savefig(p, dpi=90); plt.close(fig)
        figs.append(("J0 coupling: process_value vs J1+J2", p))

        # error histograms
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.boxplot([d["act_err"][:, i] for i in range(len(acts))], labels=acts,
                   showfliers=False)
        ax.set_ylabel("error (rad)"); ax.tick_params(axis="x", rotation=90, labelsize=6)
        fig.tight_layout()
        p = outdir / "error_hist.png"; fig.savefig(p, dpi=90); plt.close(fig)
        figs.append(("per-actuator error distribution", p))

        # coverage
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.boxplot([d["act_pos"][:, i] for i in range(len(acts))], labels=acts,
                   showfliers=False)
        ax.set_ylabel("process_value (rad)"); ax.tick_params(axis="x", rotation=90, labelsize=6)
        fig.tight_layout()
        p = outdir / "coverage.png"; fig.savefig(p, dpi=90); plt.close(fig)
        figs.append(("joint-space coverage per actuator", p))

    # summary numbers
    n_valid = int(d["valid"].sum())
    n_seg = int(d["seg_id"].max()) + 1 if (d["seg_id"] >= 0).any() else 0
    rows = [
        ("grid points", d["t"].size),
        ("valid points", n_valid),
        ("valid fraction", f"{n_valid / max(d['t'].size,1):.3f}"),
        ("segments", n_seg),
        ("dataset_rate", float(d["dataset_rate"])),
        ("duration_s", f"{t[-1]:.1f}"),
    ]
    coupling_max = {}
    jidx = {n: k for k, n in enumerate(jorder)}
    for a, (j1, j2) in coupling.items():
        ai = acts.index(a)
        summed = d["gt_pos"][:, jidx[j1]] + d["gt_pos"][:, jidx[j2]]
        coupling_max[a] = float(np.nanmax(np.abs(d["act_pos"][:, ai] - summed)))

    html = ["<html><head><meta charset='utf-8'><title>QC</title></head><body>",
            f"<h2>{ap_path.name}</h2><table border=1 cellpadding=4>"]
    for k, v in rows:
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table><h3>J0 coupling max |process_value - (J1+J2)| (rad)</h3><table border=1 cellpadding=4>")
    for a, v in coupling_max.items():
        flag = " style='background:#fdd'" if v > 0.1 else ""
        html.append(f"<tr{flag}><td>{a}</td><td>{v:.4f}</td></tr>")
    html.append("</table>")
    if not _HAVE_MPL:
        html.append("<p><b>matplotlib unavailable — figures skipped.</b></p>")
    for title, p in figs:
        html.append(f"<h3>{title}</h3><img src='data:image/png;base64,{_b64(p)}'/>")
    html.append("</body></html>")
    idx = outdir / "index.html"
    idx.write_text("\n".join(html))
    print(f"wrote {idx}  ({len(figs)} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
