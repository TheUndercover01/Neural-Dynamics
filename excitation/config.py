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
HOME_SETTLE_S = 4.0     # hold at home_pose_actuator() before every episode, see below

# --- Home / reset pose -----------------------------------------------------------------
# "Baoding-ball" default pose from roto/tasks/robots/shadowlite/shadowlite.py, vendored via
# excitation/run_simgap_hardware.py's DEFAULT_JOINT_POS (POLICY_JOINT_ORDER, raw radians).
# Independent joints map 1:1 policy<->actuator (see joints.yaml's policy_to_actuator
# comment). The 3 coupled joints (FFJ2/MFJ2/RFJ2) are DOUBLED to get the actuator's
# summed-J1+J2 command -- confirmed convention, matches run_simgap_hardware.py's
# publish_to_hand() 2x (assumes the sim's FFJ1 exactly mimics FFJ2, so summed = 2x single).
# This is the one place in the excitation package that intentionally applies that 2x --
# unlike generated trajectories (which command actuator-space radians directly and must
# never apply it), this pose STARTS in policy-space and has to cross into actuator-space.
_DEFAULT_POLICY_POS = {
    "rh_FFJ4": -0.349, "rh_MFJ4": 0.0, "rh_RFJ4": -0.349, "rh_THJ5": 0.4,
    "rh_FFJ3": 0.65, "rh_MFJ3": 0.65, "rh_RFJ3": 0.65, "rh_THJ4": 0.5,
    "rh_FFJ2": 0.87, "rh_MFJ2": 0.87, "rh_RFJ2": 0.87, "rh_THJ2": 0.35, "rh_THJ1": 0.0,
}
_COUPLED_POLICY_JOINTS = {"rh_FFJ2", "rh_MFJ2", "rh_RFJ2"}


def home_pose_actuator():
    """(13,) radians, actuator_order -- the baoding-ball default pose, coupled joints
    doubled to the actuator's summed-J1+J2 convention (see comment above), clipped to
    effective_command_limits() -- the raw values put FFJ4/RFJ4 within ~0.0001 rad of
    their literal hard mechanical stop (-0.3491), which every generated trajectory in
    this package otherwise avoids via the same 2% safety inset. Commanding a hard stop
    every single episode (this pose is held before EVERY recording) risks unnecessary
    wear, so it gets the same treatment as everything else."""
    import numpy as np
    joints = cl.load_joints()
    acts = list(joints["actuator_order"])
    p2a = dict(joints.get("policy_to_actuator", {}))
    pose = np.zeros(len(acts))
    for pj, val in _DEFAULT_POLICY_POS.items():
        act = p2a.get(pj, pj)
        pose[acts.index(act)] = val * 2.0 if pj in _COUPLED_POLICY_JOINTS else val
    lower, upper = effective_command_limits()
    return np.clip(pose, lower, upper)

# Native topic rates (informational only — rosbag has no rate/downsample setting; it
# always records every message a topic publishes. Stored in the meta/ JSON sidecar as provenance).
NATIVE_RATES_HZ = {"joint_states": 125, "controller_state": 87, "diagnostics": 2}

# --- Safety ----------------------------------------------------------------------------
# Fraction of each joint's [lower,upper] range to inset from the hard command_limits,
# so generated trajectories never command a literal mechanical stop.
SAFETY_INSET_FRAC = 0.02

# --- Rate limiting -----------------------------------------------------------------------
# Per-episode max_delta (rad/frame) is sampled from this range: gentle -> aggressive.
# Top of the range is pinned to latency.yaml's step_thresh (see compose.py) so the
# limiter can distinguish "continuous motion" from "deliberate step" by construction.
MAX_DELTA_RANGE = (0.002, 0.1)

# --- Regime -> family mix (weights need not sum to 1; compose.py normalizes) -----------
# NOTE: "free_space" always carves out >=1 channel for the `steps` family whenever
# "steps" is a key in its family dict at all (see compose.py:_allocate_channels --
# has_steps is True as soon as the key is present, independent of its weight value), so
# it can NEVER produce an all-13-channels-continuous episode by itself.
# "free_space_continuous" has no "steps" key at all -> _allocate_channels's `else`
# branch gives every one of the 13 channels to the continuous blend (ou_walk +
# multisine + chirp), simultaneously. Use this regime when you specifically want dense
# coverage of every joint moving together with no discrete jumps mixed in.
REGIME_FAMILIES = {
    "free_space":            {"ou_walk": 1.0, "multisine": 1.0, "chirp": 1.0, "steps": 0.5},
    "free_space_continuous": {"ou_walk": 1.0, "multisine": 1.0, "chirp": 1.0},
    "perturbation":          {"static_holds": 1.0, "steps": 1.0},
    "loaded_hold":           {"static_holds": 1.0, "steps": 1.0},
    "step_probe":            {"steps": 1.0},
}

# Per-episode duration (when --duration isn't given) is drawn uniformly from this range,
# same for every regime -- see run_episode.py:_build_episode(). Drawn from a Generator
# seeded with the episode's own rng_seed (not the shared/global numpy RNG), so the exact
# same seed reproduces the exact same duration too, not just the same trajectory content.
DURATION_RANGE_S = (30.0, 90.0)

# Maps the new richer `regime` to record_episode.sh's existing EXCITATION= tag
# (free_space | loaded | manual_perturbation) so latency_timeconstant.py's
# _load_tag()/by_load grouping keeps working unchanged.
REGIME_TO_EXCITATION = {
    "free_space": "free_space",
    "free_space_continuous": "free_space",
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
# false steps. Kept under 1.0 for margin (compose.py's rate limiter is a second,
# independent layer of defense on top of this). 0.65 x step_thresh(0.1) = 0.065 rad/frame
# = 3.9 rad/s at 60Hz -- close to the 4.0 rad/s thumb-joint limit, well above the 2.0
# rad/s limit most other joints have (intentional: exploring the saturation regime is
# useful actuator-net training signal, not just the achievable range).
CONTINUOUS_SLEW_FRAC = 0.65


def effective_command_limits():
    """command_limits() inset by SAFETY_INSET_FRAC on each side."""
    lower, upper = cl.command_limits()
    span = upper - lower
    inset = span * SAFETY_INSET_FRAC
    return lower + inset, upper - inset
