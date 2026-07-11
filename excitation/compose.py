"""Stitches generator families into one episode trajectory. No ROS imports — offline
testable, same as generators.py.

THE CENTRAL DESIGN CONSTRAINT (see test_generators.py's detect_steps round-trip and
CLAUDE.md's plan file for the full writeup): diagnostics/latency_timeconstant.py's
detect_steps() needs a >step_thresh jump in ONE frame followed by a tight hold for
hold_frames afterward. Two composition mistakes would break that:

  1. A global max_delta rate limit applied blindly would smear every step into a ramp
     -> "NO STEPS FOUND". Fixed by rate_limit() being STEP-AWARE: it only clamps deltas
     that are already <= step_thresh (continuous motion); a real jump passes through
     unclamped.
  2. Overlaying a continuous signal (OU/multisine/chirp) on top of a stepping channel
     would keep the jump itself detectable but violate the post-jump HOLD BAND (the
     continuous signal keeps moving after the jump), so detect_steps rejects the step
     even though it "happened". Fixed by never sharing a channel between the `steps`
     family and any continuous family — see _allocate_channels().

Continuous families that DO share channels (e.g. ou_walk + multisine + chirp all in
`free_space`) are combined with a WEIGHTED AVERAGE, not a raw sum. Each individual
family's per-frame delta is already bounded by
generators.CONTINUOUS_SLEW_FRAC * step_thresh (see generators.py); a convex combination
(weights summing to 1) of several such bounded signals is bounded by the same cap
(weighted average of magnitudes <= the shared bound). A raw sum would not preserve that
bound and could push a "continuous" channel's delta over step_thresh, which the
step-aware rate limiter would then misread as a deliberate step.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402
import generators as gen  # noqa: E402

_GENERATOR_FNS = {
    "ou_walk": gen.ou_walk,
    "multisine": gen.multisine,
    "chirp": gen.chirp,
    "static_holds": gen.static_holds,
    "steps": gen.steps,
}


def _allocate_channels(families: list[str], weights: dict, n: int, rng: np.random.Generator):
    """Split the n actuator channels between the `steps` family and the "continuous
    group" (every other requested family), proportional to normalized weight. Returns
    (steps_channels, continuous_channels) — disjoint, covering all n channels iff both
    groups are non-empty; steps_channels == range(n) if `steps` is the only family
    requested (e.g. step_probe, which should probe every joint)."""
    w = {f: float(weights.get(f, 1.0)) for f in families}
    total_w = sum(w.values()) or 1.0
    w = {f: v / total_w for f, v in w.items()}

    has_steps = "steps" in families
    continuous_families = [f for f in families if f != "steps"]

    order = list(range(n))
    rng.shuffle(order)

    if has_steps and continuous_families:
        n_steps = max(1, min(n - 1, round(w["steps"] * n)))
        steps_channels = sorted(order[:n_steps])
        continuous_channels = sorted(order[n_steps:])
    elif has_steps:
        steps_channels = list(range(n))
        continuous_channels = []
    else:
        steps_channels = []
        continuous_channels = list(range(n))
    return steps_channels, continuous_channels, w


def rate_limit(traj: np.ndarray, max_delta: float, step_thresh: float) -> np.ndarray:
    """Step-aware rate limiter. Per channel, per frame: if the INPUT trajectory's own
    |traj[t]-traj[t-1]| <= step_thresh, integrate the output by that delta clamped to
    +-max_delta (smooths continuous jitter); if the input jumps by > step_thresh, pass
    traj[t] straight through unmodified (a deliberate step — see module docstring).

    The step/continuous decision MUST be based on the input's own consecutive delta,
    never on (traj[t] - out[t-1]). Comparing against the lagged output is a trap: once
    max_delta is small enough that out falls behind traj (any "gentle" episode), the
    unclamped gap traj[t]-out[t-1] grows every frame until it exceeds step_thresh purely
    from accumulated lag, at which point the limiter "catches up" with one synthetic
    jump — fabricating a step that was never in the composed trajectory. Anchoring the
    is_step test to traj[t]-traj[t-1] (bounded by construction for continuous families,
    see generators.py CONTINUOUS_SLEW_FRAC) makes that impossible: continuous channels
    can never trigger it, no matter how far behind the smoothed output lags.
    """
    assert max_delta <= step_thresh, (
        f"max_delta ({max_delta}) must be <= step_thresh ({step_thresh}), otherwise "
        "clamped continuous motion could itself exceed step_thresh and be misread as a step")
    T, n = traj.shape
    out = traj.copy()
    for t in range(1, T):
        raw_delta = traj[t] - traj[t - 1]
        is_step = np.abs(raw_delta) > step_thresh
        clamped = np.clip(raw_delta, -max_delta, max_delta)
        out[t] = np.where(is_step, traj[t], out[t - 1] + clamped)
    return out


def compose_episode(regime: str, families: list[str], weights: dict, *, rng_seed: int,
                     duration_s: float, rate: float | None = None, joint_limits=None,
                     max_delta: float | None = None):
    """Generate + combine `families` into one (T,13) episode.

    Returns (traj, step_events, info) where:
      traj        (T,13) float radians, actuator_order, clipped + rate-limited.
      step_events list of {frame, actuator_idx, pre, post, magnitude} — ground truth
                  from the `steps` family only (empty if `steps` not in `families`).
      info        dict: family_seeds, channel allocation, max_delta used, and
                  per-channel delta max/p95 (meta/ JSON provenance — lets you filter
                  to a real max_delta scale later without recollecting).
    """
    rate = rate or ecfg.PUBLISH_RATE_HZ
    lower, upper = joint_limits if joint_limits is not None else ecfg.effective_command_limits()
    n = len(lower)
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)

    steps_channels, continuous_channels, w = _allocate_channels(families, weights, n, rng)
    family_seeds = {f: int(rng.integers(0, 2**31 - 1)) for f in families}

    midpoint = 0.5 * (lower + upper)
    traj = np.tile(midpoint, (T, 1))

    step_events: list[dict] = []
    if "steps" in families and steps_channels:
        steps_traj = gen.steps(duration_s, (lower, upper), family_seeds["steps"],
                                rate=rate, channels=steps_channels)
        traj[:, steps_channels] = steps_traj[:, steps_channels]
        step_events = gen.step_schedule(duration_s, (lower, upper), family_seeds["steps"],
                                         rate=rate, channels=steps_channels)

    continuous_families = [f for f in families if f != "steps"]
    if continuous_channels and continuous_families:
        cont_w_total = sum(w[f] for f in continuous_families) or 1.0
        blend = np.zeros((T, n))
        for f in continuous_families:
            fam_traj = _GENERATOR_FNS[f](duration_s, (lower, upper), family_seeds[f],
                                          rate=rate, channels=continuous_channels)
            blend += (w[f] / cont_w_total) * fam_traj
        traj[:, continuous_channels] = blend[:, continuous_channels]

    traj = np.clip(traj, lower, upper)

    step_thresh = float(cl.load_latency()["step_thresh"])
    if max_delta is None:
        max_delta = float(rng.uniform(*ecfg.MAX_DELTA_RANGE))
    max_delta = min(max_delta, step_thresh)  # enforce the rate_limit() precondition
    traj = rate_limit(traj, max_delta, step_thresh)

    delta = np.abs(np.diff(traj, axis=0)) if T > 1 else np.zeros((0, n))
    info = {
        "regime": regime,
        "families": list(families),
        "family_seeds": family_seeds,
        "steps_channels": steps_channels,
        "continuous_channels": continuous_channels,
        "max_delta_rad": max_delta,
        "per_channel_delta_max": delta.max(axis=0).tolist() if delta.size else [0.0] * n,
        "per_channel_delta_p95": (np.percentile(delta, 95, axis=0).tolist()
                                   if delta.size else [0.0] * n),
    }
    return traj, step_events, info
