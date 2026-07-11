#!/usr/bin/env python3
"""Continuous, slow bidirectional sweep of each coupled J0 actuator (FFJ0/MFJ0/RFJ0), to
test the tendon-slack/backlash hypothesis: does J2 "unlock" (start moving) EARLIER than the
ideal ~100 deg (1.745 rad) handoff point while descending from full flexion (pi) toward
extension (0), and if so, does that early unlock show up as a torque anomaly -- or is
torque essentially insensitive to which joint is moving, since one motor drives both
(config/joints.yaml `coupling`)?

Command convention (verified live this session): the J0 controller's /command IS the raw
summed J1+J2 angle, range 0..pi (NOT the excitation/run_simgap_hardware.py 2x policy path --
that path is for the RL policy's 0..pi/2 per-joint convention, unrelated to this test).

Sweep style: CONTINUOUS, not step-dwell-sample. The commanded sum ramps linearly at
SWEEP_SPEED_RAD_S (slow enough that friction/backlash dominates over dynamic/damping
torque) while every /joint_states + J0 controller /state message that arrives is logged
with its real timestamp and the commanded value in effect at that instant. This is a
classic hysteresis-loop measurement: the down-vs-up gap at the same commanded value (or
the same measured joint angle) IS the backlash.

Safety: assumes the hand has ALREADY been moved to a non-colliding pose (thumb clear of the
index -- confirmed visually by the operator; FF/MF/RF spread via J4 abduction). This script
additionally holds every OTHER actuator fixed at HOLD_POSE for the whole run and re-asserts
it between fingers, but does NOT move the thumb (verify thumb clearance yourself first).

Writes outputs/coupled_backlash_sweep.md (verdict + per-finger tables) and
outputs/coupled_backlash_<finger>.png:
  (a) J1 & J2 measured angle vs COMMANDED sum, down/up overlaid (position hysteresis loop)
  (b) effort vs COMMANDED sum, down/up (torque vs command)
  (c) effort vs MEASURED joint angle, separately for J1 and J2 over each one's own active
      band, down/up (torque vs the actual moving joint's own angle)

    python3 scripts/coupled_backlash_sweep.py
    FINGERS=rh_FFJ0 SWEEP_SPEED_RAD_S=0.15 N_PASSES=3 python3 scripts/coupled_backlash_sweep.py
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

import numpy as np  # noqa: E402
# rospy/ROS msg imports are deliberately lazy (inside the functions that need them), not
# at module top -- mirrors preprocess/parse_bag.py and scripts/check_stream.py's convention
# so the RAW_DIR offline re-analysis path (see main()) runs on a machine with no ROS at all.

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

# dataviz palette -- same constants as diagnostics/latency_timeconstant.py, kept in sync
# manually since this module intentionally doesn't import that one (no coupling needed).
_BLUE = "#2a78d6"
_AQUA = "#1baf7a"
_RED = "#e34948"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_INK = "#0b0b0b"

EPS = float(os.environ.get("EPS", "0.05"))                       # inset from 0/pi hard stops
SWEEP_SPEED_RAD_S = float(os.environ.get("SWEEP_SPEED_RAD_S", "0.1"))  # slow -> quasi-static
PUBLISH_RATE_HZ = float(os.environ.get("PUBLISH_RATE_HZ", "30"))
N_PASSES = int(os.environ.get("N_PASSES", "3"))
N_GRID = int(os.environ.get("N_GRID", "120"))  # interpolation grid points for analysis/plots
IDEAL_BREAKPOINT = 1.745  # rad (~100 deg) -- user's nominal J1/J2 handoff point, reference only
ACTIVE_SLOPE_THRESH = 0.3  # rad/rad -- |dJ/dcmd| above this counts as "this joint is moving"

# Whole-hand hold pose while one finger sweeps (mirrors the manually-verified safe pose:
# thumb de-opposed+clear, FF/MF/RF spread via J4, J3 mid-flex). Thumb is NOT re-sent here --
# confirm it's clear yourself before running.
HOLD_POSE = {
    "ffj0": 0.87, "ffj3": 0.65, "ffj4": -0.349,
    "mfj0": 0.87, "mfj3": 0.65, "mfj4": 0.0,
    "rfj0": 0.87, "rfj3": 0.65, "rfj4": -0.349,
}


def goto_hold_pose(pubs: dict, hold_sec: float = 2.0) -> None:
    import rospy
    from std_msgs.msg import Float64
    for act, val in HOLD_POSE.items():
        pubs[act].publish(Float64(val))
    rospy.sleep(hold_sec)


def sweep_finger_continuous(actuator: str, j1: str, j2: str, joints: dict, pubs: dict,
                             topics: dict) -> np.ndarray:
    """Continuously ramp the summed setpoint pi-EPS <-> EPS, N_PASSES times, logging every
    /joint_states + controller /state message with the commanded value in effect at that
    instant. Returns an (N, 9) array: t, cmd, j1, j2, vel, effort, pwm, pass_idx, dir_code
    (dir_code: 0=down, 1=up)."""
    import rospy
    from control_msgs.msg import JointControllerState
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64
    jorder = list(joints["joint_order"])
    j1i, j2i = jorder.index(j1), jorder.index(j2)
    pub = pubs[actuator[3:].lower()]  # rh_FFJ0 -> ffj0

    shared = {"cmd": None, "pass_idx": 0, "dir_code": 0}
    state_last = {"command": float("nan")}
    records = []

    def on_js(msg: JointState) -> None:
        if shared["cmd"] is None:
            return
        records.append((msg.header.stamp.to_sec(), shared["cmd"], msg.position[j1i],
                         msg.position[j2i], msg.velocity[j1i], msg.effort[j1i],
                         state_last["command"], shared["pass_idx"], shared["dir_code"]))

    def on_state(msg: JointControllerState) -> None:
        state_last["command"] = msg.command

    sub = rospy.Subscriber(topics["joint_states"], JointState, on_js, queue_size=400)
    state_sub = rospy.Subscriber(cl.controller_state_topic(actuator), JointControllerState,
                                  on_state, queue_size=20)
    rospy.sleep(0.3)  # let subscribers connect

    hi, lo = np.pi - EPS, EPS
    duration_s = abs(hi - lo) / SWEEP_SPEED_RAD_S
    n_ticks = max(2, int(round(duration_s * PUBLISH_RATE_HZ)))
    rate = rospy.Rate(PUBLISH_RATE_HZ)

    for p in range(N_PASSES):
        for dir_code, (start, end) in ((0, (hi, lo)), (1, (lo, hi))):
            shared["pass_idx"], shared["dir_code"] = p, dir_code
            for tgt in np.linspace(start, end, n_ticks):
                shared["cmd"] = float(tgt)
                pub.publish(Float64(float(tgt)))
                rate.sleep()

    sub.unregister()
    state_sub.unregister()
    return np.asarray(records, dtype=float)


def _interp_pass_avg(arr: np.ndarray, dir_code: int, grid: np.ndarray) -> dict:
    """Interpolate J1/J2/effort/pwm onto `grid` (commanded-value axis) per pass, then
    average + std across passes -- continuous analogue of the old discrete grid-average."""
    n_passes = int(arr[:, 7].max()) + 1 if arr.size else 0
    j1_p, j2_p, eff_p, pwm_p = [], [], [], []
    for p in range(n_passes):
        sel = arr[(arr[:, 7] == p) & (arr[:, 8] == dir_code)]
        if sel.shape[0] < 2:
            continue
        order = np.argsort(sel[:, 1])  # sort by cmd (monotonic in time within one leg)
        cmd_s, j1_s, j2_s, eff_s, pwm_s = (sel[order, 1], sel[order, 2], sel[order, 3],
                                           sel[order, 5], sel[order, 6])
        j1_p.append(np.interp(grid, cmd_s, j1_s))
        j2_p.append(np.interp(grid, cmd_s, j2_s))
        eff_p.append(np.interp(grid, cmd_s, eff_s))
        finite = np.isfinite(pwm_s)
        pwm_p.append(np.interp(grid, cmd_s, pwm_s) if finite.any()
                     else np.full_like(grid, np.nan))
    if not j1_p:
        return {"cmd": grid, "j1": np.full_like(grid, np.nan), "j2": np.full_like(grid, np.nan),
                "j1_std": np.full_like(grid, np.nan), "j2_std": np.full_like(grid, np.nan),
                "effort": np.full_like(grid, np.nan), "pwm": np.full_like(grid, np.nan)}
    return {
        "cmd": grid, "j1": np.mean(j1_p, axis=0), "j2": np.mean(j2_p, axis=0),
        "j1_std": np.std(j1_p, axis=0), "j2_std": np.std(j2_p, axis=0),
        "effort": np.mean(eff_p, axis=0), "pwm": np.nanmean(pwm_p, axis=0),
    }


def _slopes(cmd: np.ndarray, val: np.ndarray) -> np.ndarray:
    return np.gradient(val, cmd)


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Light moving-average -- suppresses single-point spikes (e.g. the stiction/backlash
    'snap' at each sweep direction-reversal) before thresholding for activity."""
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _find_active_edge(active_mask: np.ndarray, from_low: bool,
                       min_run_frac: float = 0.1) -> int | None:
    """Boundary between a joint's contiguous active region and the SUSTAINED inactive
    region beyond it (>= min_run_frac * N consecutive inactive points), scanning from one
    end. Requiring a sustained run (not just the next point) is what makes this robust to
    an isolated boundary/reversal-transient spike being mistaken for real activity."""
    n = active_mask.size
    min_run = max(3, int(round(min_run_frac * n)))
    order = np.arange(n) if from_low else np.arange(n)[::-1]
    run = 0
    last_active_idx = None
    for k in order:
        if active_mask[k]:
            run = 0
            last_active_idx = k
        else:
            run += 1
            if run >= min_run and last_active_idx is not None:
                return last_active_idx
    return last_active_idx


BOUNDARY_TRIM_FRAC = 0.03  # exclude this fraction of grid points at each end from
                           # active-region detection -- contaminated by the direction-
                           # reversal stiction/backlash "snap", not the J1/J2 handoff


def analyze_finger(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"status": "no_data"}

    lo, hi = EPS, np.pi - EPS
    grid = np.linspace(lo, hi, N_GRID)  # ascending, for clean gradients
    down = _interp_pass_avg(arr, dir_code=0, grid=grid)
    up = _interp_pass_avg(arr, dir_code=1, grid=grid)

    # --- kinematic split + early-unlock, from the DESCENDING sweep (matches user's framing:
    # 180->100 j1 moves, 100->0 j2 moves). grid is ascending cmd already. Slope is smoothed
    # and boundary points are excluded before thresholding -- see _smooth/_find_active_edge.
    slope_j1 = _smooth(np.abs(_slopes(grid, down["j1"])))
    slope_j2 = _smooth(np.abs(_slopes(grid, down["j2"])))
    j2_active = slope_j2 > ACTIVE_SLOPE_THRESH
    j1_active = slope_j1 > ACTIVE_SLOPE_THRESH
    trim = max(1, int(round(BOUNDARY_TRIM_FRAC * N_GRID)))
    j2_active[:trim] = False
    j2_active[-trim:] = False
    j1_active[:trim] = False
    j1_active[-trim:] = False

    # J2 is active starting from the LOW-cmd end (per observed kinematics: it moves first
    # as cmd rises from 0) -- find where that contiguous active run ends.
    j2_edge_idx = _find_active_edge(j2_active, from_low=True)
    # J1 is active starting from the HIGH-cmd end -- find where ITS contiguous run ends.
    j1_edge_idx = _find_active_edge(j1_active, from_low=False)

    j2_unlock_cmd = float(grid[j2_edge_idx]) if j2_edge_idx is not None else None
    j1_active_range = (float(grid[j1_edge_idx]), float(grid[-trim - 1])) \
        if j1_edge_idx is not None else None
    j2_active_range = (float(grid[trim]), float(grid[j2_edge_idx])) \
        if j2_edge_idx is not None else None
    early_by = (j2_unlock_cmd - IDEAL_BREAKPOINT) if j2_unlock_cmd is not None else None

    # --- backlash / hysteresis: down-up gap at matched COMMANDED value, on the shared grid --
    hyst_j1 = down["j1"] - up["j1"]
    hyst_j2 = down["j2"] - up["j2"]

    # --- torque feature near the unlock point (vs commanded value) -------------------------
    torque_note = "n/a"
    if j2_unlock_cmd is not None:
        near = np.abs(grid - j2_unlock_cmd) < (3 * (grid[1] - grid[0]))
        far = ~near
        if near.sum() >= 2 and far.sum() >= 2:
            slope_near = float(np.abs(np.gradient(down["effort"][near], grid[near])).mean())
            slope_far = float(np.abs(np.gradient(down["effort"][far], grid[far])).mean())
            ratio = slope_near / slope_far if slope_far > 1e-9 else float("inf")
            torque_note = (f"|d(effort)/d(cmd)| near unlock = {slope_near:.2f}, "
                           f"elsewhere = {slope_far:.2f} (ratio {ratio:.2f}x)")

    return {
        "status": "ok", "down": down, "up": up, "arr": arr,
        "j1_active_range": j1_active_range, "j2_active_range": j2_active_range,
        "j2_unlock_cmd": j2_unlock_cmd, "early_by": early_by,
        "hyst_j1_mean": float(np.nanmean(np.abs(hyst_j1))),
        "hyst_j1_max": float(np.nanmax(np.abs(hyst_j1))),
        "hyst_j2_mean": float(np.nanmean(np.abs(hyst_j2))),
        "hyst_j2_max": float(np.nanmax(np.abs(hyst_j2))),
        "torque_note": torque_note, "n_samples": int(arr.shape[0]),
    }


def plot_finger(finger: str, analysis: dict, outdir: pathlib.Path) -> pathlib.Path | None:
    if not _HAVE_MPL or analysis["status"] != "ok":
        return None
    down, up, arr = analysis["down"], analysis["up"], analysis["arr"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9.5))

    # (a) position hysteresis vs COMMANDED sum
    ax1.plot(down["cmd"], down["j1"], color=_BLUE, lw=2, label="J1 (descend)")
    ax1.plot(up["cmd"], up["j1"], color=_BLUE, lw=1.2, ls="--", label="J1 (ascend)")
    ax1.plot(down["cmd"], down["j2"], color=_AQUA, lw=2, label="J2 (descend)")
    ax1.plot(up["cmd"], up["j2"], color=_AQUA, lw=1.2, ls="--", label="J2 (ascend)")
    ax1.axvline(IDEAL_BREAKPOINT, color=_MUTED, ls=":", lw=1, label="ideal 100deg")
    if analysis["j2_unlock_cmd"] is not None:
        ax1.axvline(analysis["j2_unlock_cmd"], color=_RED, ls=":", lw=1.2,
                    label="measured J2 unlock")
    ax1.set_xlabel("commanded sum J1+J2 (rad)")
    ax1.set_ylabel("joint angle (rad)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title(f"{finger}: position hysteresis loop (down vs up)", fontsize=10)

    # (b) torque vs COMMANDED sum
    ax2.plot(down["cmd"], down["effort"], color=_BLUE, lw=2, label="effort (descend)")
    ax2.plot(up["cmd"], up["effort"], color=_BLUE, lw=1.2, ls="--", label="effort (ascend)")
    ax2.axvline(IDEAL_BREAKPOINT, color=_MUTED, ls=":", lw=1)
    if analysis["j2_unlock_cmd"] is not None:
        ax2.axvline(analysis["j2_unlock_cmd"], color=_RED, ls=":", lw=1.2)
    ax2.set_xlabel("commanded sum J1+J2 (rad)")
    ax2.set_ylabel("effort")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_title(f"{finger}: torque vs COMMANDED value", fontsize=10)

    # (c) torque vs MEASURED joint angle -- raw scatter (faint) + per-joint active-band line
    raw_down = arr[arr[:, 8] == 0]
    raw_up = arr[arr[:, 8] == 1]
    ax3.scatter(raw_down[:, 2], raw_down[:, 5], s=3, alpha=0.15, color=_BLUE)
    ax3.scatter(raw_up[:, 2], raw_up[:, 5], s=3, alpha=0.15, color=_MUTED)
    ax3.scatter(raw_down[:, 3], raw_down[:, 5], s=3, alpha=0.15, color=_AQUA)
    ax3.scatter(raw_up[:, 3], raw_up[:, 5], s=3, alpha=0.15, color=_MUTED)
    j1r, j2r = analysis["j1_active_range"], analysis["j2_active_range"]
    if j1r:
        m = (down["j1"] >= j1r[0]) & (down["j1"] <= j1r[1])
        ax3.plot(down["j1"][m], down["effort"][m], color=_BLUE, lw=2, label="effort vs J1")
    if j2r:
        m = (down["j2"] >= j2r[0]) & (down["j2"] <= j2r[1])
        ax3.plot(down["j2"][m], down["effort"][m], color=_AQUA, lw=2, label="effort vs J2")
    ax3.set_xlabel("measured joint angle (rad) -- J1 (blue) / J2 (aqua)")
    ax3.set_ylabel("effort")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_title(f"{finger}: torque vs MEASURED joint angle (raw scatter + descend mean)",
                  fontsize=10)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=_GRID, lw=0.6)

    fig.tight_layout()
    p = outdir / f"coupled_backlash_{finger}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def render_md(results: dict) -> str:
    lines = ["# Coupled-joint tendon-slack / backlash sweep", ""]
    lines.append(
        f"Continuous bidirectional sweep of each J0 actuator's summed setpoint "
        f"(pi-{EPS:g} <-> {EPS:g}) at {SWEEP_SPEED_RAD_S:g} rad/s, {N_PASSES} passes -- a "
        "classic slow hysteresis-loop measurement, not stop-and-hold static sampling. Every "
        "arriving `/joint_states` sample is logged continuously with the commanded value in "
        "effect at that instant; analysis interpolates each pass onto a common commanded-"
        f"value grid ({N_GRID} points) and averages across passes. Reference breakpoint: the "
        f"user's nominal J1/J2 handoff at {IDEAL_BREAKPOINT:g} rad (~100 deg) -- descending "
        "from pi, J1 should move alone down to this point, then J2 alone the rest of the way "
        "to 0. The measured handoff point can land EARLY (J2 starts moving while cmd is still "
        "above the reference) or LATE (J1 keeps moving past the reference, J2 stays locked "
        "longer) -- both are reported as measured, not assumed.")
    lines.append("")

    for finger, a in results.items():
        lines.append(f"## {finger}")
        if a["status"] != "ok":
            lines.append(f"- status: **{a['status']}**")
            lines.append("")
            continue
        j1r, j2r = a["j1_active_range"], a["j2_active_range"]
        lines.append(f"- samples logged: {a['n_samples']}")
        lines.append(f"- J1 active band (descending): "
                      f"{f'{j1r[0]:.3f} - {j1r[1]:.3f} rad' if j1r else 'not detected'}")
        lines.append(f"- J2 active band (descending): "
                      f"{f'{j2r[0]:.3f} - {j2r[1]:.3f} rad' if j2r else 'not detected'}")
        if a["j2_unlock_cmd"] is not None:
            early = a["early_by"]  # positive = unlocks at a HIGHER cmd (sooner in the
                                    # descent, i.e. EARLY); negative = LOWER cmd (LATE)
            if early > 0.03:
                verdict = (f"EARLY by {early:.3f} rad ({np.degrees(early):.1f} deg) -- J2 "
                           f"starts moving while cmd is still ABOVE the {IDEAL_BREAKPOINT:g} "
                           "rad reference (still early in the descent from pi)")
            elif early < -0.03:
                verdict = (f"LATE by {abs(early):.3f} rad ({np.degrees(abs(early)):.1f} deg) "
                           f"-- J1 keeps moving PAST the {IDEAL_BREAKPOINT:g} rad reference; "
                           "J2 doesn't start until cmd has dropped further, below it")
            else:
                verdict = "matches the reference closely -- no measurable early/late unlock"
            lines.append(f"- J2 unlock point (descending): {a['j2_unlock_cmd']:.3f} rad "
                         f"-> **{verdict}**")
        else:
            lines.append("- J2 unlock point: not detected (no slope crossed the active "
                         "threshold -- check ACTIVE_SLOPE_THRESH or raw data)")
        lines.append(f"- Backlash (hysteresis, down-up at matched commanded value): "
                     f"J1 mean={a['hyst_j1_mean']:.4f} max={a['hyst_j1_max']:.4f} rad; "
                     f"J2 mean={a['hyst_j2_mean']:.4f} max={a['hyst_j2_max']:.4f} rad")
        lines.append(f"- Torque feature near unlock: {a['torque_note']}")
        lines.append("")

    lines.append("## Hypothesis verdict")
    notes = []
    for finger, a in results.items():
        if a["status"] != "ok" or "ratio" not in a["torque_note"]:
            continue
        try:
            ratio = float(a["torque_note"].split("ratio ")[1].split("x")[0])
        except (IndexError, ValueError):
            continue
        notes.append((finger, ratio))
    if notes:
        max_ratio = max(r for _, r in notes)
        if max_ratio < 2.0:
            lines.append(
                f"Effort/torque slope near the J2-unlock point is within ~{max_ratio:.1f}x "
                "of the slope elsewhere on the sweep, across all fingers -- **supports the "
                "hypothesis**: since one motor drives both J1 and J2, the measured torque "
                "does not show a distinct feature at the handoff/unlock point, even where "
                "the measured handoff deviates from the ideal 100 deg reference.")
        else:
            lines.append(
                f"Effort/torque slope near the J2-unlock point is up to {max_ratio:.1f}x "
                "steeper than elsewhere -- **against the hypothesis**: torque DOES show a "
                "measurable feature at the handoff, worth a closer look at which finger/pass.")
    else:
        lines.append("Insufficient torque-feature data to render a verdict "
                      "(see per-finger notes above).")
    return "\n".join(lines) + "\n"


def main() -> int:
    joints = cl.load_joints()
    topics = cl.load_topics()
    coupling = cl.coupled_actuators(joints)  # {rh_FFJ0: [rh_FFJ1, rh_FFJ2], ...}

    only = os.environ.get("FINGERS")
    if only:
        wanted = set(only.split(","))
        coupling = {k: v for k, v in coupling.items() if k in wanted}

    outdir = cl.REPO_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # RAW_DIR: re-analyze previously-saved raw sweeps offline, no hardware/ROS needed --
    # every sweep's raw (t, cmd, j1, j2, vel, effort, pwm, pass, dir) array is saved below,
    # specifically so an analysis-only bug fix never requires re-running the physical hand.
    raw_dir_override = os.environ.get("RAW_DIR")
    results = {}
    if raw_dir_override:
        raw_dir = pathlib.Path(raw_dir_override)
        for actuator in coupling:
            raw_path = raw_dir / f"coupled_backlash_{actuator}_raw.npz"
            if not raw_path.exists():
                print(f"skip {actuator}: {raw_path} not found", file=sys.stderr)
                continue
            arr = np.load(raw_path)["arr"]
            results[actuator] = analyze_finger(arr)
            print(f"   re-analyzed {actuator} from {raw_path}, n={arr.shape[0]}")
    else:
        import rospy
        from std_msgs.msg import Float64
        rospy.init_node("coupled_backlash_sweep", anonymous=True, disable_signals=True)

        all_acts = ["ffj0", "ffj3", "ffj4", "mfj0", "mfj3", "mfj4", "rfj0", "rfj3", "rfj4"]
        pubs = {a: rospy.Publisher(cl.controller_command_topic(f"rh_{a.upper()}"), Float64,
                                   queue_size=1) for a in all_acts}
        rospy.sleep(0.5)
        print("-- asserting hold pose --")
        goto_hold_pose(pubs)

        for actuator, (j1, j2) in coupling.items():
            print(f"-- sweeping {actuator} ({j1}/{j2}): {N_PASSES} continuous bidirectional "
                  f"passes @ {SWEEP_SPEED_RAD_S:g} rad/s --")
            arr = sweep_finger_continuous(actuator, j1, j2, joints, pubs, topics)
            np.savez_compressed(outdir / f"coupled_backlash_{actuator}_raw.npz", arr=arr)
            results[actuator] = analyze_finger(arr)
            print(f"   {results[actuator].get('status')}, n={arr.shape[0]} "
                  f"(raw saved to coupled_backlash_{actuator}_raw.npz)")
            goto_hold_pose(pubs, hold_sec=1.0)  # restore this finger to hold before the next

    (outdir / "coupled_backlash_sweep.md").write_text(render_md(results))
    print(f"wrote {outdir / 'coupled_backlash_sweep.md'}")

    for finger, a in results.items():
        p = plot_finger(finger, a, outdir)
        if p:
            print(f"wrote {p}")
    if not _HAVE_MPL:
        print("matplotlib unavailable -- skipped PNGs", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
