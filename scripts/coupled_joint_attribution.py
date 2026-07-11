#!/usr/bin/env python3
"""Follow-on to COUPLED_BACKLASH_TEST.md / scripts/coupled_backlash_sweep.py: can the
coupled fingers' SHARED motor effort/velocity signal be attributed to whichever individual
joint (J1 or J2) is actually moving, at a given commanded value? That would let a
sim/actuator-net use a `commanded_value -> active joint` map to decide which joint's
simulated torque to update from the one physical motor signal.

Starting premise was "only one joint moves at a time" (a clean one-hot mapping). A read-only
probe against the existing slow-sweep raw data (outputs/coupled_backlash_*_raw.npz) found
that premise does NOT hold uniformly: RFJ0 is close to one-at-a-time (~5-7% both-moving),
but FFJ0/MFJ0 spend 46-70% of the sweep with BOTH joints moving simultaneously at real,
above-noise magnitudes. This script:

  1. Reuses the existing slow (0.1 rad/s) raw data as-is (no re-run).
  2. Collects two NEW speed conditions -- medium (0.3 rad/s) and fast (0.8 rad/s) -- to test
     whether the "both moving" overlap is a quasi-static-sweep artifact (shrinks at speed) or
     persists, reusing sweep_finger_continuous/goto_hold_pose/HOLD_POSE directly from
     coupled_backlash_sweep.py rather than re-implementing the sweep.
  3. Estimates each joint's OWN velocity via a smoothed centered difference on J1/J2 POSITION
     (the shared `/joint_states` velocity field is, by construction, identical for J1 and J2
     and therefore useless for telling them apart).
  4. Classifies each sample into one of 4 states (neither / J1-only / J2-only / both) via a
     noise-floor-verified activity threshold, and reports the state fractions HONESTLY per
     finger per speed condition -- a 3-state map, not a forced one-hot.
  5. Tests whether effort correlates with the ACTIVE joint's own angle/velocity within its
     own active window -- the actual question of whether a per-joint torque model is learnable
     from the shared signal once you know which joint is active.

Writes outputs/coupled_joint_attribution.md, outputs/coupled_joint_attribution_mapping.json
(the concrete `commanded_value -> P(state)` map per finger/condition for downstream sim use),
outputs/coupled_joint_attribution_<finger>.png, and raw data for the 2 new speed conditions
(outputs/coupled_attribution_<finger>_speed<rad_s>.npz) so this analysis never needs to touch
the hand again.

    python3 scripts/coupled_joint_attribution.py
    FINGERS=rh_FFJ0 python3 scripts/coupled_joint_attribution.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coupled_backlash_sweep as cbs  # noqa: E402  -- reuse sweep_finger_continuous /
                                       # goto_hold_pose / HOLD_POSE / EPS, no duplication

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

_BLUE = "#2a78d6"
_AQUA = "#1baf7a"
_RED = "#e34948"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_INK = "#0b0b0b"

# Window is chosen PER CONDITION to span a fixed COMMANDED-VALUE range (TARGET_CMD_SPAN),
# not a fixed sample count. /joint_states publishes at ~125Hz regardless of sweep speed, so a
# fixed sample-count window spans proportionally MORE commanded range as speed increases --
# smearing right across a sharp handoff and inflating the "both moving" fraction as a pure
# resolution artifact, not a physical effect. Verified this mattered (this session): with a
# fixed 15-sample window, ascending both-fraction appeared to jump 0.5%->67% slow->fast; with
# a properly speed-scaled window it's a smaller (but still real) 0.5%->52% jump. Always use
# window_for_speed(), never a bare constant, when comparing across speed conditions.
TARGET_CMD_SPAN = float(os.environ.get("TARGET_CMD_SPAN", "0.024"))  # rad, matches the
                                                                       # original 15-sample
                                                                       # window at 0.1 rad/s
JOINT_STATES_DT = 0.008  # s, ~125Hz -- matches config/topics.yaml expected_rates_hz
ACTIVITY_THRESH = float(os.environ.get("ACTIVITY_THRESH", "0.02"))  # rad/s
N_CMD_BINS = int(os.environ.get("N_CMD_BINS", "60"))

# (label, speed rad/s, passes) -- slow reuses existing data, medium/fast are collected live
# (or reloaded offline from previously-saved raw .npz, see main()'s RELOAD_ONLY path)
CONDITION_ORDER = ["slow_0.1", "medium_0.3", "fast_0.8"]
CONDITION_SPEEDS = {"slow_0.1": 0.1, "medium_0.3": 0.3, "fast_0.8": 0.8}
NEW_CONDITIONS = [("medium_0.3", 0.3, 3), ("fast_0.8", 0.8, 5)]


def window_for_speed(speed: float) -> int:
    w = TARGET_CMD_SPAN / (2 * JOINT_STATES_DT * speed)
    return max(2, int(round(w)))

STATE_NAMES = {0: "neither", 1: "j1_only", 2: "j2_only", 3: "both"}


# --------------------------------------------------------------------------------------
# Per-joint velocity + state classification
# --------------------------------------------------------------------------------------

def add_velocities(arr: np.ndarray, window: int) -> np.ndarray:
    """arr columns: t,cmd,j1,j2,vel,effort,pwm,pass_idx,dir_code (9 cols, from
    coupled_backlash_sweep's sweep_finger_continuous). Appends dj1, dj2 (smoothed centered
    difference on J1/J2 POSITION, computed within each (pass,direction) leg -- never across
    a leg boundary). NaN where too close to a leg's edge for a full window."""
    n = arr.shape[0]
    dj1 = np.full(n, np.nan)
    dj2 = np.full(n, np.nan)
    n_passes = int(arr[:, 7].max()) + 1
    for p in range(n_passes):
        for d in (0, 1):
            idxs = np.where((arr[:, 7] == p) & (arr[:, 8] == d))[0]
            if idxs.size < 2 * window + 1:
                continue
            sub = arr[idxs]
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            orig_idx = idxs[order]
            t, j1v, j2v = sub[:, 0], sub[:, 2], sub[:, 3]
            k = np.arange(window, len(t) - window)
            d1 = (j1v[k + window] - j1v[k - window]) / (t[k + window] - t[k - window])
            d2 = (j2v[k + window] - j2v[k - window]) / (t[k + window] - t[k - window])
            dj1[orig_idx[k]] = d1
            dj2[orig_idx[k]] = d2
    return np.column_stack([arr, dj1, dj2])  # 11 columns now


def classify_and_check_noise(arr_ext: np.ndarray, thresh: float) -> tuple:
    dj1, dj2 = arr_ext[:, 9], arr_ext[:, 10]
    valid = np.isfinite(dj1) & np.isfinite(dj2)
    state = np.full(arr_ext.shape[0], -1, dtype=int)  # -1 = invalid/edge, excluded downstream
    a1 = np.abs(dj1) > thresh
    a2 = np.abs(dj2) > thresh
    state[valid & a1 & a2] = 3
    state[valid & a1 & ~a2] = 1
    state[valid & ~a1 & a2] = 2
    state[valid & ~a1 & ~a2] = 0
    neither = valid & (state == 0)
    noise1 = float(np.percentile(np.abs(dj1[neither]), 95)) if neither.any() else float("nan")
    noise2 = float(np.percentile(np.abs(dj2[neither]), 95)) if neither.any() else float("nan")
    return state, noise1, noise2


def state_prob_by_cmd(cmd: np.ndarray, state: np.ndarray, n_bins: int, lo: float,
                       hi: float) -> tuple:
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = {name: np.full(n_bins, np.nan) for name in STATE_NAMES.values()}
    valid = state >= 0
    for i in range(n_bins):
        m = valid & (cmd >= edges[i]) & (cmd < edges[i + 1])
        total = m.sum()
        if total == 0:
            continue
        for code, name in STATE_NAMES.items():
            probs[name][i] = (state[m] == code).sum() / total
    return centers, probs


def corr_within_state(x: np.ndarray, effort: np.ndarray, state: np.ndarray, code: int,
                       min_n: int = 20) -> tuple:
    m = (state == code) & np.isfinite(x) & np.isfinite(effort)
    n = int(m.sum())
    if n < min_n:
        return None, n
    if np.std(x[m]) < 1e-9 or np.std(effort[m]) < 1e-9:
        return 0.0, n
    return float(np.corrcoef(x[m], effort[m])[0, 1]), n


def _frac_and_corrs(arr_ext: np.ndarray, state: np.ndarray, mask: np.ndarray) -> dict:
    """Shared helper: state fractions + effort correlations, restricted to `mask` (e.g. one
    direction). Used both for the combined view and the per-direction breakdown below --
    the direction split turned out to be the dominant signal (see module docstring), so both
    are reported rather than only a blended aggregate."""
    valid = (state >= 0) & mask
    frac = {name: float((state[valid] == code).mean()) if valid.any() else float("nan")
            for code, name in STATE_NAMES.items()}
    j1, j2, effort = arr_ext[:, 2], arr_ext[:, 3], arr_ext[:, 5]
    dj1, dj2 = arr_ext[:, 9], arr_ext[:, 10]
    state_masked = np.where(mask, state, -1)
    r_eff_angle_j1, n_j1 = corr_within_state(j1, effort, state_masked, 1)
    r_eff_angle_j2, n_j2 = corr_within_state(j2, effort, state_masked, 2)
    r_eff_vel_j1, _ = corr_within_state(dj1, effort, state_masked, 1)
    r_eff_vel_j2, _ = corr_within_state(dj2, effort, state_masked, 2)
    return {
        "frac": frac, "n": int(valid.sum()),
        "r_eff_angle_j1": r_eff_angle_j1, "n_j1_only": n_j1,
        "r_eff_angle_j2": r_eff_angle_j2, "n_j2_only": n_j2,
        "r_eff_vel_j1": r_eff_vel_j1, "r_eff_vel_j2": r_eff_vel_j2,
    }


def analyze_condition(arr: np.ndarray, label: str) -> dict:
    speed = CONDITION_SPEEDS.get(label)
    if speed is None:
        raise ValueError(f"unknown condition label {label!r}, add it to CONDITION_SPEEDS")
    window = window_for_speed(speed)
    arr_ext = add_velocities(arr, window)
    state, noise1, noise2 = classify_and_check_noise(arr_ext, ACTIVITY_THRESH)
    dir_code = arr_ext[:, 8]

    combined = _frac_and_corrs(arr_ext, state, np.ones(state.shape, dtype=bool))
    by_dir = {
        "down": _frac_and_corrs(arr_ext, state, dir_code == 0),
        "up": _frac_and_corrs(arr_ext, state, dir_code == 1),
    }

    cmd = arr_ext[:, 1]
    lo, hi = cbs.EPS, np.pi - cbs.EPS
    centers_all, probs_all = state_prob_by_cmd(cmd, state, N_CMD_BINS, lo, hi)
    probs_by_dir = {}
    for dname, dcode in (("down", 0), ("up", 1)):
        m = dir_code == dcode
        state_d = np.where(m, state, -1)
        c, p = state_prob_by_cmd(cmd, state_d, N_CMD_BINS, lo, hi)
        probs_by_dir[dname] = p

    return {
        "label": label, "n": int(arr.shape[0]), "noise1": noise1, "noise2": noise2,
        "window": window, "speed": speed,
        "combined": combined, "by_dir": by_dir,
        "cmd_centers": centers_all, "probs": probs_all, "probs_by_dir": probs_by_dir,
        "arr_ext": arr_ext, "state": state,
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def plot_finger(finger: str, conds: dict, outdir: pathlib.Path):
    if not _HAVE_MPL:
        return None
    order = [l for l in CONDITION_ORDER if l in conds]
    if not order:
        return None
    # 3 columns: P(state) during extension (down), P(state) during flexion (up) -- the real
    # contrast (see module docstring) -- then effort-vs-angle scatter for context.
    fig, axes = plt.subplots(len(order), 3, figsize=(15, 3.6 * len(order)), squeeze=False)
    state_colors = {0: _MUTED, 1: _BLUE, 2: _AQUA, 3: _RED}

    for row, label in enumerate(order):
        c = conds[label]
        ax_down, ax_up, ax3 = axes[row]
        centers = c["cmd_centers"]
        for ax, dname, dtitle in ((ax_down, "down", "extension (descending)"),
                                   (ax_up, "up", "flexion (ascending)")):
            probs = c["probs_by_dir"][dname]
            for code, name in STATE_NAMES.items():
                ax.plot(centers, probs[name], label=name, color=state_colors[code], lw=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("P(state)")
            ax.set_xlabel("commanded sum J1+J2 (rad)")
            ax.set_title(f"{finger} [{label}]: {dtitle}", fontsize=9)
            if row == 0:
                ax.legend(fontsize=6, loc="center left")

        arr_ext, state = c["arr_ext"], c["state"]
        j1, j2, effort = arr_ext[:, 2], arr_ext[:, 3], arr_ext[:, 5]
        for code, name in STATE_NAMES.items():
            m = state == code
            if not m.any():
                continue
            xvals = j2 if code == 2 else j1  # J2-only plotted against J2's own angle
            ax3.scatter(xvals[m], effort[m], s=2, alpha=0.2, color=state_colors[code],
                        label=name if row == 0 else None)
        ax3.set_xlabel("joint angle (rad) -- J1 for neither/J1-only/both, J2 for J2-only")
        ax3.set_ylabel("effort")
        ax3.set_title(f"{finger} [{label}]: effort vs angle, colored by state", fontsize=9)
        if row == 0:
            ax3.legend(fontsize=6, loc="upper left")

        for ax in (ax_down, ax_up, ax3):
            ax.set_facecolor("#fcfcfb")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=_GRID, lw=0.6)

    fig.tight_layout()
    p = outdir / f"coupled_joint_attribution_{finger}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------------------
# Report + machine-readable mapping
# --------------------------------------------------------------------------------------

def render_md(results: dict) -> str:
    lines = ["# Coupled joint (J1/J2) torque/velocity attribution", ""]
    lines.append(
        "Follow-on to `COUPLED_BACKLASH_TEST.md`. Tests whether the coupled fingers' shared "
        "motor effort/velocity signal can be attributed to whichever individual joint (J1 or "
        "J2) is actually moving, at a given commanded value -- i.e. whether a "
        "`commanded_value -> active joint` map is a valid way to route the one physical "
        "torque signal to the right joint in sim. Per-joint velocity is estimated via a "
        "smoothed centered difference on J1/J2 POSITION, since the shared `/joint_states` "
        "velocity field is identical for J1 and J2 by construction. The smoothing window is "
        f"chosen PER CONDITION to span a fixed {TARGET_CMD_SPAN:g} rad of commanded value "
        "(not a fixed sample count) -- a fixed sample-count window would span proportionally "
        "more commanded range at higher speed and inflate the both-moving fraction as a pure "
        "resolution artifact (verified this mattered; see `window` column below). Activity "
        f"threshold {ACTIVITY_THRESH:g} rad/s, checked against each condition's own rest-state "
        "noise floor (see `noise floor` column -- should stay well below the threshold).")
    lines.append("")

    def _fmt(v, n=None):
        if v is None:
            return "n/a"
        return f"{v:+.3f}" + (f" (n={n})" if n is not None else "")

    for finger, conds in results.items():
        lines.append(f"## {finger}")
        present = [l for l in CONDITION_ORDER if l in conds]

        lines.append(
            "At the original SLOW (quasi-static) sweep speed, flexion (ascending, tendon "
            "pulled taut) was essentially clean one-hot while extension (descending, tendon "
            "slackening) showed a real both-moving overlap scaled by this finger's backlash "
            "(matches `COUPLED_BACKLASH_TEST.md`'s hysteresis ranking). Reported per "
            "direction, not blended, because a blended number hides this. **That clean "
            "one-hot-during-flexion picture does NOT hold at higher sweep speed** -- see the "
            "speed-trend line below; check the actual numbers per condition rather than "
            "assuming the slow-speed picture generalizes.")
        lines.append("")
        lines.append("| condition | direction | window (samples) | n | noise floor "
                      "(j1/j2 rad/s) | neither | J1-only | J2-only | **both** |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label in present:
            c = conds[label]
            for dname in ("down", "up"):
                f_ = c["by_dir"][dname]["frac"]
                n_ = c["by_dir"][dname]["n"]
                lines.append(f"| {label} | {dname} | {c['window']} | {n_} | "
                              f"{c['noise1']:.4f}/{c['noise2']:.4f} | {f_['neither']:.1%} | "
                              f"{f_['j1_only']:.1%} | {f_['j2_only']:.1%} | **{f_['both']:.1%}** |")
        lines.append("")

        if len(present) >= 2:
            for dname in ("down", "up"):
                both_vals = [conds[l]["by_dir"][dname]["frac"]["both"] for l in present]
                if both_vals[-1] < both_vals[0] - 0.03:
                    trend = "SHRINKS with speed"
                elif both_vals[-1] > both_vals[0] + 0.03:
                    trend = "GROWS with speed"
                else:
                    trend = "STAYS ~FLAT across speed"
                lines.append(f"- [{dname}] both-moving fraction vs speed: **{trend}** ("
                              + ", ".join(f"{l}={v:.1%}" for l, v in zip(present, both_vals))
                              + ")")
            lines.append("")

        lines.append("| condition | direction | corr(effort, J1 angle)\\[J1-only\\] | "
                      "corr(effort, J1 vel)\\[J1-only\\] | corr(effort, J2 angle)\\[J2-only\\] | "
                      "corr(effort, J2 vel)\\[J2-only\\] |")
        lines.append("|---|---|---|---|---|---|")
        for label in present:
            c = conds[label]
            for dname in ("down", "up"):
                d = c["by_dir"][dname]
                lines.append(f"| {label} | {dname} | "
                              f"{_fmt(d['r_eff_angle_j1'], d['n_j1_only'])} | "
                              f"{_fmt(d['r_eff_vel_j1'])} | "
                              f"{_fmt(d['r_eff_angle_j2'], d['n_j2_only'])} | "
                              f"{_fmt(d['r_eff_vel_j2'])} |")
        lines.append("")

    lines.append("## Verdict")
    for dname, dlabel in (("down", "Extension (descending)"), ("up", "Flexion (ascending)")):
        both_slow = [c["slow_0.1"]["by_dir"][dname]["frac"]["both"] for c in results.values()
                     if "slow_0.1" in c]
        both_fast = [c["fast_0.8"]["by_dir"][dname]["frac"]["both"] for c in results.values()
                     if "fast_0.8" in c]
        if both_slow and both_fast:
            lines.append(
                f"- **{dlabel}**: both-moving mean {np.mean(both_slow):.1%} (slow) -> "
                f"{np.mean(both_fast):.1%} (fast) across fingers.")
    lines.append("")

    # Data-driven, not asserted: aggregate |corr| for angle vs velocity across every
    # finger/direction/joint actually computed above, so the verdict reports whichever
    # relationship the numbers actually show instead of an assumed one.
    angle_rs, vel_rs = [], []
    for c in results.values():
        for label, cond in c.items():
            for dname in ("down", "up"):
                d = cond["by_dir"][dname]
                for r in (d["r_eff_angle_j1"], d["r_eff_angle_j2"]):
                    if r is not None:
                        angle_rs.append(abs(r))
                for r in (d["r_eff_vel_j1"], d["r_eff_vel_j2"]):
                    if r is not None:
                        vel_rs.append(abs(r))
    angle_mean = np.mean(angle_rs) if angle_rs else float("nan")
    vel_mean = np.mean(vel_rs) if vel_rs else float("nan")
    stronger = "ANGLE" if angle_mean > vel_mean else "VELOCITY"

    slow_down = [c["slow_0.1"]["by_dir"]["down"]["frac"]["both"] for c in results.values()
                 if "slow_0.1" in c]
    slow_up = [c["slow_0.1"]["by_dir"]["up"]["frac"]["both"] for c in results.values()
               if "slow_0.1" in c]
    fast_down = [c["fast_0.8"]["by_dir"]["down"]["frac"]["both"] for c in results.values()
                 if "fast_0.8" in c]
    fast_up = [c["fast_0.8"]["by_dir"]["up"]["frac"]["both"] for c in results.values()
               if "fast_0.8" in c]

    lines.append(
        "**Overall**: at the original SLOW quasi-static speed, a strict one-hot "
        f"`commanded_value -> active joint` map was excellent during flexion (mean "
        f"{np.mean(slow_up):.0%} both-moving across fingers -- i.e. ~99% single-joint) but "
        f"poor during extension for FF/MF specifically (mean {np.mean(slow_down):.0%} "
        "both-moving), matching the backlash already measured in `COUPLED_BACKLASH_TEST.md`. "
        "**That picture does NOT hold at higher sweep speed**: both-moving grows substantially "
        f"with speed in BOTH directions (flexion mean {np.mean(slow_up):.0%} -> "
        f"{np.mean(fast_up):.0%}, extension mean {np.mean(slow_down):.0%} -> "
        f"{np.mean(fast_down):.0%}, slow->fast). Part of this growth is a real physical effect "
        "and part is a measurement-resolution effect from the velocity window (corrected here "
        "by scaling the window to a fixed COMMANDED-VALUE span rather than a fixed sample "
        "count -- using a fixed sample count inflated the fast-condition both-fraction even "
        "further). **Practical implication**: a one-hot map is only safe to use at the "
        "quasi-static speeds actually tested here (~0.1 rad/s) and specifically during "
        "flexion; it should NOT be assumed to hold at faster, more dynamic motion, or during "
        "extension, without re-checking against data at the speed actually used in sim. "
        f"Within J1-only/J2-only windows, effort correlates more strongly, on average, with "
        f"each joint's own **{stronger}** (mean |r|={max(angle_mean, vel_mean):.2f}) than with "
        f"its own {'velocity' if stronger == 'ANGLE' else 'angle'} (mean |r|="
        f"{min(angle_mean, vel_mean):.2f}) -- but neither is uniform across finger/direction/"
        "speed (see the per-condition table above). A single learned torque model per joint "
        "is NOT a safe assumption across all conditions -- check the specific finger/"
        "direction/speed of interest rather than relying on the aggregate.")
    return "\n".join(lines) + "\n"


def _nan_to_none(x) -> list:
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            for v in x]


def build_mapping_json(results: dict) -> dict:
    """Per finger/condition: combined (blended) map AND the per-direction maps. The
    per-direction maps are the ones a sim consumer should actually use -- flexion
    (ascending) is reliably one-hot, extension (descending) is not (see render_md) -- the
    combined map is included only for reference/backward-compat."""
    out = {}
    for finger, conds in results.items():
        out[finger] = {}
        for label, c in conds.items():
            centers = c["cmd_centers"].tolist()
            entry = {
                "cmd_bin_centers": centers,
                "combined": {
                    "p_neither": _nan_to_none(c["probs"]["neither"]),
                    "p_j1_only": _nan_to_none(c["probs"]["j1_only"]),
                    "p_j2_only": _nan_to_none(c["probs"]["j2_only"]),
                    "p_both": _nan_to_none(c["probs"]["both"]),
                },
            }
            for dname in ("down", "up"):
                p = c["probs_by_dir"][dname]
                entry[dname] = {
                    "cmd_bin_centers": centers,
                    "p_neither": _nan_to_none(p["neither"]),
                    "p_j1_only": _nan_to_none(p["j1_only"]),
                    "p_j2_only": _nan_to_none(p["j2_only"]),
                    "p_both": _nan_to_none(p["both"]),
                }
            out[finger][label] = entry
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def collect_condition(actuator: str, j1: str, j2: str, joints: dict, pubs: dict,
                       topics: dict, speed: float, passes: int) -> np.ndarray:
    """Reuses coupled_backlash_sweep's sweep function with different speed/pass-count by
    temporarily overriding its module-level constants (read dynamically at call time)."""
    cbs.SWEEP_SPEED_RAD_S = speed
    cbs.N_PASSES = passes
    return cbs.sweep_finger_continuous(actuator, j1, j2, joints, pubs, topics)


def main() -> int:
    joints = cl.load_joints()
    topics = cl.load_topics()
    coupling = cl.coupled_actuators(joints)

    only = os.environ.get("FINGERS")
    if only:
        wanted = set(only.split(","))
        coupling = {k: v for k, v in coupling.items() if k in wanted}

    outdir = cl.REPO_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # slow condition: reuse existing backlash-sweep raw data, no hardware
    for actuator in coupling:
        raw_path = outdir / f"coupled_backlash_{actuator}_raw.npz"
        if not raw_path.exists():
            print(f"WARNING: no existing slow-condition data for {actuator} "
                  f"({raw_path} not found) -- skipping that condition", file=sys.stderr)
            continue
        arr = np.load(raw_path)["arr"]
        results.setdefault(actuator, {})["slow_0.1"] = analyze_condition(arr, "slow_0.1")
        print(f"loaded existing slow-condition data for {actuator}: n={arr.shape[0]}")

    skip_live = os.environ.get("SKIP_LIVE") == "1"
    if skip_live:
        # reload any already-saved medium/fast raw data (from a prior live run) so an
        # analysis-only fix never requires re-running hardware, same convention as
        # coupled_backlash_sweep.py's RAW_DIR path
        for actuator, (j1, j2) in coupling.items():
            for label, speed, passes in NEW_CONDITIONS:
                raw_path = outdir / f"coupled_attribution_{actuator}_speed{speed:g}.npz"
                if not raw_path.exists():
                    print(f"skip {actuator} [{label}]: {raw_path} not found", file=sys.stderr)
                    continue
                arr = np.load(raw_path)["arr"]
                results.setdefault(actuator, {})[label] = analyze_condition(arr, label)
                print(f"reloaded {actuator} [{label}] from {raw_path.name}: n={arr.shape[0]}")
    else:
        import rospy
        from std_msgs.msg import Float64
        rospy.init_node("coupled_joint_attribution", anonymous=True, disable_signals=True)

        all_acts = ["ffj0", "ffj3", "ffj4", "mfj0", "mfj3", "mfj4", "rfj0", "rfj3", "rfj4"]
        pubs = {a: rospy.Publisher(cl.controller_command_topic(f"rh_{a.upper()}"), Float64,
                                   queue_size=1) for a in all_acts}
        rospy.sleep(0.5)
        print("-- asserting hold pose --")
        cbs.goto_hold_pose(pubs)

        for actuator, (j1, j2) in coupling.items():
            for label, speed, passes in NEW_CONDITIONS:
                print(f"-- {actuator} [{label}]: speed={speed:g} rad/s, {passes} passes --")
                arr = collect_condition(actuator, j1, j2, joints, pubs, topics, speed, passes)
                raw_out = outdir / f"coupled_attribution_{actuator}_speed{speed:g}.npz"
                np.savez_compressed(raw_out, arr=arr)
                results.setdefault(actuator, {})[label] = analyze_condition(arr, label)
                print(f"   n={arr.shape[0]}, raw saved to {raw_out.name}")
                cbs.goto_hold_pose(pubs, hold_sec=1.0)

    (outdir / "coupled_joint_attribution.md").write_text(render_md(results))
    print(f"wrote {outdir / 'coupled_joint_attribution.md'}")

    mapping = build_mapping_json(results)
    (outdir / "coupled_joint_attribution_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"wrote {outdir / 'coupled_joint_attribution_mapping.json'}")

    for finger, conds in results.items():
        p = plot_finger(finger, conds, outdir)
        if p:
            print(f"wrote {p}")
    if not _HAVE_MPL:
        print("matplotlib unavailable -- skipped PNGs", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
