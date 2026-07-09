"""Excitation-only tunables. Canonical layout (actuator_order, command_limits, topics)
lives in config_lib.py / config/joints.yaml — this module imports that, never redeclares it.

Usage from any subdir:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import config_lib as cl
    from excitation import config as ecfg   # or same-dir `import config as ecfg`
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

# --- Timing --------------------------------------------------------------------------
PUBLISH_RATE_HZ = 60.0
DT = 1.0 / PUBLISH_RATE_HZ
LEAD_IN_S = 2.0     # hold/ramp before the generated trajectory starts (let rosbag subscribe)
LEAD_OUT_S = 2.0    # hold after the trajectory ends (recording margin)
RECORD_MARGIN_S = 3.0   # extra rosbag duration beyond lead-in/out + trajectory, safety buffer

# Native topic rates (informational only — rosbag has no rate/downsample setting; it
# always records every message a topic publishes. Stored in meta.yaml as provenance).
NATIVE_RATES_HZ = {"joint_states": 125, "controller_state": 87, "diagnostics": 2}

# --- Safety ----------------------------------------------------------------------------
# Fraction of each joint's [lower,upper] range to inset from the hard command_limits,
# so generated trajectories never command a literal mechanical stop.
SAFETY_INSET_FRAC = 0.02

# --- Rate limiting -----------------------------------------------------------------------
# Per-episode max_delta (rad/frame) is sampled from this range: gentle -> aggressive.
# Top of the range is pinned to latency.yaml's step_thresh (see compose.py) so the
# limiter can distinguish "continuous motion" from "deliberate step" by construction.
MAX_DELTA_RANGE = (0.002, 0.05)

# --- Regime -> family mix (weights need not sum to 1; compose.py normalizes) -----------
REGIME_FAMILIES = {
    "free_space":   {"ou_walk": 1.0, "multisine": 1.0, "chirp": 1.0, "steps": 0.5},
    "perturbation": {"static_holds": 1.0, "steps": 1.0},
    "loaded_hold":  {"static_holds": 1.0, "steps": 1.0},
    "step_probe":   {"steps": 1.0},
}

REGIME_DEFAULT_DURATION_S = {
    "free_space": 40.0,
    "perturbation": 30.0,
    "loaded_hold": 30.0,
    "step_probe": 20.0,
}

# Maps the new richer `regime` to record_episode.sh's existing EXCITATION= tag
# (free_space | loaded | manual_perturbation) so latency_timeconstant.py's
# _load_tag()/by_load grouping keeps working unchanged.
REGIME_TO_EXCITATION = {
    "free_space": "free_space",
    "perturbation": "manual_perturbation",
    "loaded_hold": "loaded",
    "step_probe": "free_space",
}

# --- Generator parameter bounds (randomized per-episode within these) ------------------
OU_THETA_RANGE = (0.05, 2.0)          # mean-reversion rate, 1/s
OU_SIGMA_FRAC_RANGE = (0.02, 0.15)    # noise std, fraction of joint half-range
MULTISINE_N_COMPONENTS_RANGE = (3, 6)
MULTISINE_FREQ_HZ_RANGE = (0.05, 1.5)
MULTISINE_AMP_FRAC_RANGE = (0.1, 0.8)      # sum of component amplitudes, fraction of half-range
CHIRP_FREQ_HZ_RANGE = (0.05, 2.0)
CHIRP_AMP_FRAC_RANGE = (0.3, 0.9)
STEP_HOLD_S_RANGE = (0.8, 1.5)
STEP_MAGNITUDE_FRAC_RANGE = (0.15, 0.8)     # fraction of joint range, guaranteed > step_thresh
STATIC_HOLD_DWELL_S_RANGE = (1.0, 4.0)

# Fraction of latency.yaml's step_thresh that continuous families (ou_walk, multisine,
# chirp) cap their per-frame |delta| to, BY CONSTRUCTION. Needed because naive parameter
# draws (e.g. OU mean-reversion on a wide-range joint like FFJ0, or several summed sine
# components) can exceed step_thresh purely from the math, which would trip
# diagnostics/latency_timeconstant.py's detect_steps() and pollute latency stats with
# false steps. Kept well under 1.0 for margin (compose.py's rate limiter is a second,
# independent layer of defense on top of this).
CONTINUOUS_SLEW_FRAC = 0.4


def effective_command_limits():
    """command_limits() inset by SAFETY_INSET_FRAC on each side."""
    lower, upper = cl.command_limits()
    span = upper - lower
    inset = span * SAFETY_INSET_FRAC
    return lower + inset, upper - inset
