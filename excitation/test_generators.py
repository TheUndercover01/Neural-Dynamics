#!/usr/bin/env python3
"""Offline correctness gate for excitation/generators.py (and compose.py, once added).

No ROS required. Runnable-script style like qc/loader_test.py: prints PASS/FAIL per
check, nonzero exit on any failure. matplotlib is optional (plots skipped, not fatal,
if unavailable).

    python3 excitation/test_generators.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))       # same dir: config.py, generators.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # repo root: config_lib.py, diagnostics/
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402
import generators as gen  # noqa: E402
import compose  # noqa: E402
from diagnostics.latency_timeconstant import detect_steps  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

DURATION_S = 30.0
SEEDS = [0, 1, 2, 3, 4]

CONTINUOUS_FAMILIES = {
    "ou_walk": gen.ou_walk,
    "multisine": gen.multisine,
    "chirp": gen.chirp,
    "static_holds": gen.static_holds,
}
ALL_FAMILIES = {**CONTINUOUS_FAMILIES, "steps": gen.steps}

_failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(label)


def main() -> int:
    joints = cl.load_joints()
    lower, upper = cl.command_limits()
    limits = (lower, upper)
    step_thresh = float(cl.load_latency()["step_thresh"])
    hold_frames = int(cl.load_latency()["hold_frames"])
    rate = ecfg.PUBLISH_RATE_HZ
    n = len(joints["actuator_order"])

    outdir = cl.REPO_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------
    # 1. Shapes, no NaNs
    # ---------------------------------------------------------------------------------
    for name, fn in ALL_FAMILIES.items():
        traj = fn(DURATION_S, limits, rng_seed=0)
        expected_T = round(DURATION_S * rate)
        _check(f"shape: {name} is ({expected_T},{n})",
               traj.shape == (expected_T, n), f"got {traj.shape}")
        _check(f"no NaNs: {name}", not np.isnan(traj).any())

    # ---------------------------------------------------------------------------------
    # 2. Within command_limits
    # ---------------------------------------------------------------------------------
    eps = 1e-9
    for name, fn in ALL_FAMILIES.items():
        traj = fn(DURATION_S, limits, rng_seed=1)
        within = bool(np.all((traj >= lower - eps) & (traj <= upper + eps)))
        _check(f"within command_limits: {name}", within)

    # ---------------------------------------------------------------------------------
    # 3. Continuous families never exceed step_thresh per frame (by construction)
    # ---------------------------------------------------------------------------------
    for name, fn in CONTINUOUS_FAMILIES.items():
        worst = 0.0
        for seed in SEEDS:
            traj = fn(DURATION_S, limits, rng_seed=seed)
            worst = max(worst, float(np.abs(np.diff(traj, axis=0)).max()))
        _check(f"max |delta| < step_thresh: {name} ({worst:.4f} < {step_thresh})",
               worst < step_thresh)

    # `steps` must actually produce jumps bigger than step_thresh (that's its job).
    steps_traj = gen.steps(DURATION_S, limits, rng_seed=2)
    steps_worst = float(np.abs(np.diff(steps_traj, axis=0)).max())
    _check(f"steps produces jumps > step_thresh ({steps_worst:.4f} > {step_thresh})",
           steps_worst > step_thresh)

    # ---------------------------------------------------------------------------------
    # 4. Coverage: each channel spans a healthy fraction of its range
    # ---------------------------------------------------------------------------------
    combined = np.concatenate(
        [fn(DURATION_S, limits, rng_seed=s) for fn in ALL_FAMILIES.values() for s in SEEDS], axis=0)
    span = upper - lower
    coverage_frac = (combined.max(axis=0) - combined.min(axis=0)) / np.where(span > 0, span, 1.0)
    low_coverage = [joints["actuator_order"][i] for i in range(n) if coverage_frac[i] < 0.4]
    _check("combined families cover >=40% of range on every channel",
           len(low_coverage) == 0, f"under 40%: {low_coverage}")

    # ---------------------------------------------------------------------------------
    # 5. Multi-joint simultaneity: default (channels=None) calls move several joints at once
    # ---------------------------------------------------------------------------------
    for name, fn in CONTINUOUS_FAMILIES.items():
        traj = fn(10.0, limits, rng_seed=3)
        moved = np.abs(traj - traj[0]).max(axis=0) > 1e-6
        _check(f"multi-joint: {name} moves >=4 channels simultaneously",
               int(moved.sum()) >= 4, f"only {int(moved.sum())} moved")

    # ---------------------------------------------------------------------------------
    # 6. False-positive guard: detect_steps() must find ~nothing in continuous families
    # ---------------------------------------------------------------------------------
    for name, fn in CONTINUOUS_FAMILIES.items():
        total_detected = 0
        for seed in SEEDS:
            traj = fn(DURATION_S, limits, rng_seed=seed)
            seg_id = np.zeros(traj.shape[0], dtype=int)
            for ch in range(n):
                total_detected += len(detect_steps(traj[:, ch], seg_id, step_thresh, hold_frames))
        _check(f"detect_steps finds 0 false-positive steps in {name} (across {len(SEEDS)} seeds)",
               total_detected == 0, f"found {total_detected}")

    # ---------------------------------------------------------------------------------
    # 7. Positive guard: detect_steps recovers exactly steps()'s own step_schedule
    # ---------------------------------------------------------------------------------
    step_channels = [0, 3, 7, 9]
    sched = gen.step_schedule(DURATION_S, limits, rng_seed=4, channels=step_channels)
    steps_traj2 = gen.steps(DURATION_S, limits, rng_seed=4, channels=step_channels)
    seg_id = np.zeros(steps_traj2.shape[0], dtype=int)
    expected_by_ch: dict[int, list[int]] = {}
    for e in sched:
        expected_by_ch.setdefault(e["actuator_idx"], []).append(e["frame"])
    all_match = True
    for ch in step_channels:
        detected = detect_steps(steps_traj2[:, ch], seg_id, step_thresh, hold_frames)
        detected_frames = sorted(d["step_frame"] for d in detected)
        expected_frames = sorted(expected_by_ch.get(ch, []))
        if detected_frames != expected_frames:
            all_match = False
            print(f"    channel {ch}: expected {expected_frames}, detected {detected_frames}")
    _check("detect_steps recovers exact step_schedule() frames", all_match)

    # ---------------------------------------------------------------------------------
    # 8. compose_episode round-trip: the correctness contract from compose.py's module
    #    docstring — step_probe recovers exactly its composed step_events; the other
    #    regimes yield ~zero detected steps (continuous families must not leak steps
    #    onto shared channels, and steps-family channels must not get false extras).
    # ---------------------------------------------------------------------------------
    # Swept over several seeds, not just one: an earlier version of this test passed at
    # seed=7 by chance while a rate_limit() lag-accumulation bug (fixed) was fabricating
    # false-positive steps at other seeds — a single fixed seed is not a reliable gate
    # for a stochastic composition path.
    COMPOSE_SEEDS = [0, 2, 4, 6, 7, 8, 11, 12, 16, 17, 20, 22]
    for regime, families in ecfg.REGIME_FAMILIES.items():
        fam_names = list(families.keys())
        regime_step_mismatches = 0
        regime_false_positives = 0
        regime_out_of_limits = 0
        for seed in COMPOSE_SEEDS:
            traj, events, info = compose.compose_episode(
                regime, fam_names, families, rng_seed=seed, duration_s=DURATION_S)
            seg_id = np.zeros(traj.shape[0], dtype=int)

            expected_by_ch: dict[int, list[int]] = {}
            for e in events:
                expected_by_ch.setdefault(e["actuator_idx"], []).append(e["frame"])

            detected_by_ch: dict[int, list[int]] = {}
            for ch in range(n):
                detected = detect_steps(traj[:, ch], seg_id, step_thresh, hold_frames)
                detected_by_ch[ch] = sorted(d["step_frame"] for d in detected)

            steps_ch = set(info["steps_channels"])
            cont_ch = set(info["continuous_channels"])
            for ch in steps_ch:
                if detected_by_ch[ch] != sorted(expected_by_ch.get(ch, [])):
                    regime_step_mismatches += 1
            for ch in cont_ch:
                regime_false_positives += len(detected_by_ch[ch])
            if not bool(np.all((traj >= lower - eps) & (traj <= upper + eps))):
                regime_out_of_limits += 1

        _check(f"compose[{regime}]: steps-channel detections match step_events "
               f"(across {len(COMPOSE_SEEDS)} seeds)",
               regime_step_mismatches == 0, f"{regime_step_mismatches} mismatched channel(s)")
        _check(f"compose[{regime}]: 0 false-positive steps on continuous channels "
               f"(across {len(COMPOSE_SEEDS)} seeds)",
               regime_false_positives == 0, f"found {regime_false_positives}")
        _check(f"compose[{regime}]: traj within command_limits (across {len(COMPOSE_SEEDS)} seeds)",
               regime_out_of_limits == 0, f"{regime_out_of_limits} seed(s) out of limits")

    # ---------------------------------------------------------------------------------
    # 9. Plots (best-effort, not fatal)
    # ---------------------------------------------------------------------------------
    n_plots = 0
    if _HAVE_MPL:
        for name, fn in ALL_FAMILIES.items():
            traj = fn(10.0, limits, rng_seed=0)
            fig, ax = plt.subplots(figsize=(8, 3))
            t = np.arange(traj.shape[0]) / rate
            for ch in range(n):
                ax.plot(t, traj[:, ch], lw=0.8, alpha=0.7)
            ax.set_title(f"excitation family: {name}")
            ax.set_xlabel("s"); ax.set_ylabel("rad")
            fig.tight_layout()
            p = outdir / f"excitation_{name}.png"
            fig.savefig(p, dpi=100)
            plt.close(fig)
            n_plots += 1
        print(f"wrote {n_plots} plot(s) to {outdir}")
    else:
        print("matplotlib unavailable — skipping plots")

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        return 1
    print("All checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
