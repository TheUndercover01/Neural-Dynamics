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

Continuous families that share the SAME channel group (e.g. ou_walk + multisine + chirp
all in `free_space`) each get their OWN EXCLUSIVE channels within it (see
_allocate_channels()'s continuous sub-partition) rather than being blended together via
weighted average on the same channels. An earlier version did blend them, and measured
range coverage on real hardware was badly diluted by it (a weighted/convex combination
of several signals that aren't in phase is mathematically smaller than any one of them
alone -- confirmed: even a "fast" episode only covered 16-30% of most joints' true
range). Giving each family exclusive channels means its own amplitude budget
(excitation.config's *_FRAC_RANGE constants) is never diluted by another family.
compose_episode() also now randomizes a per-channel, per-episode anchor point
(excitation.config.CENTER_FRAC_RANGE) instead of every family always orbiting the exact
joint midpoint, so different episodes explore different regions of the range instead of
all wiggling around the identical center.
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
    "sweep": gen.sweep,
}


def _allocate_channels(families: list[str], weights: dict, n: int, rng: np.random.Generator,
                        active_channels: list[int] | None = None):
    """Split the n actuator channels between the `steps` family and the "continuous
    group" (every other requested family), proportional to normalized weight. Returns
    (steps_channels, continuous_channels, family_channels, w):
      steps_channels, continuous_channels — disjoint, covering all n channels iff both
        groups are non-empty; steps_channels == range(n) if `steps` is the only family
        requested (e.g. step_probe, which should probe every joint).
      family_channels — {family: [channel indices]} further sub-partitioning
        continuous_channels among the individual continuous families (ou_walk/
        multisine/chirp), proportional to the same normalized weights, so each channel
        is driven EXCLUSIVELY by one family (see module docstring for why blending them
        instead diluted range coverage). Uses a largest-remainder allocation, with a
        small random tiebreak (so an exact tie -- e.g. 1 pool channel split across 3
        equally-weighted families -- doesn't always favor the same family by list
        order), so integer rounding never systematically starves any family of every
        channel. Empty dict if there are no continuous families requested.

    `active_channels`, if given, restricts the WHOLE pool to just those channel indices
    (e.g. excitation/run_episode.py's --joint for the single_joint regime) -- every
    other channel gets neither steps nor a continuous family, i.e. stays wherever
    compose_episode()'s inactive_value put it, for the entire episode.
    """
    w = {f: float(weights.get(f, 1.0)) for f in families}
    total_w = sum(w.values()) or 1.0
    w = {f: v / total_w for f, v in w.items()}

    has_steps = "steps" in families
    continuous_families = [f for f in families if f != "steps"]

    order = list(range(n)) if active_channels is None else list(active_channels)
    rng.shuffle(order)
    m = len(order)

    if has_steps and continuous_families and m > 1:
        n_steps = max(1, min(m - 1, round(w["steps"] * m)))
        steps_channels = sorted(order[:n_steps])
        continuous_pool = order[n_steps:]  # keep shuffled for the sub-partition below
    elif has_steps and not continuous_families:
        steps_channels = list(order)
        continuous_pool = []
    elif has_steps:
        # m == 1 with both families requested: can't split a single channel into two
        # disjoint non-empty groups. Give it to steps -- callers that need a single
        # active channel to also show continuous behavior should omit "steps" from
        # that regime's family mix instead (see config.py's single_joint comment).
        steps_channels = list(order)
        continuous_pool = []
    else:
        steps_channels = []
        continuous_pool = order

    continuous_channels = sorted(continuous_pool)

    family_channels: dict[str, list[int]] = {f: [] for f in continuous_families}
    if continuous_families and continuous_pool:
        cont_w_total = sum(w[f] for f in continuous_families) or 1.0
        norm_w = [w[f] / cont_w_total for f in continuous_families]
        n_pool = len(continuous_pool)
        raw_counts = [nw * n_pool for nw in norm_w]
        counts = [int(rc) for rc in raw_counts]
        remainder = n_pool - sum(counts)
        tiebreak = rng.random(len(continuous_families))
        by_frac_desc = sorted(range(len(continuous_families)),
                               key=lambda i: (raw_counts[i] - counts[i], tiebreak[i]),
                               reverse=True)
        for i in by_frac_desc[:remainder]:
            counts[i] += 1
        idx = 0
        for f, c in zip(continuous_families, counts):
            family_channels[f] = sorted(continuous_pool[idx:idx + c])
            idx += c

    return steps_channels, continuous_channels, family_channels, w


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
                     max_delta: float | None = None, active_channels: list[int] | None = None,
                     inactive_value=None):
    """Generate + combine `families` into one (T,13) episode.

    `active_channels` (e.g. run_episode.py's --joint) restricts which channels can be
    driven by ANY family at all -- every other channel stays at `inactive_value` (default:
    joint midpoint) for the whole episode. Used by the single_joint regime to isolate one
    actuator; `inactive_value` should be excitation.config.home_pose_actuator() there so
    the "held" channels genuinely don't move from wherever the episode's own home-reset
    already put them, rather than drifting to an arbitrary midpoint.

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

    steps_channels, continuous_channels, family_channels, w = _allocate_channels(
        families, weights, n, rng, active_channels=active_channels)
    family_seeds = {f: int(rng.integers(0, 2**31 - 1)) for f in families}

    midpoint = 0.5 * (lower + upper)
    base = midpoint if inactive_value is None else np.broadcast_to(inactive_value, (n,)).astype(float)
    traj = np.tile(base, (T, 1))

    step_events: list[dict] = []
    if "steps" in families and steps_channels:
        steps_traj = gen.steps(duration_s, (lower, upper), family_seeds["steps"],
                                rate=rate, channels=steps_channels)
        traj[:, steps_channels] = steps_traj[:, steps_channels]
        step_events = gen.step_schedule(duration_s, (lower, upper), family_seeds["steps"],
                                         rate=rate, channels=steps_channels)

    # Per-episode, per-channel anchor point for the continuous families -- randomized
    # instead of always the exact midpoint (see module docstring / excitation/config.py's
    # CENTER_FRAC_RANGE comment for why: a fixed midpoint anchor meant every episode
    # wiggled around the identical center regardless of seed). Drawn once here (not
    # per-family) so if multiple continuous families ever shared a channel again in the
    # future they'd still agree on where "center" is.
    center = lower + rng.uniform(*ecfg.CENTER_FRAC_RANGE, size=n) * (upper - lower)

    continuous_families = [f for f in families if f != "steps"]
    for f in continuous_families:
        fam_channels = family_channels.get(f, [])
        if not fam_channels:
            continue
        kwargs = {"rate": rate, "channels": fam_channels}
        if f == "ou_walk":
            kwargs["mu"] = center
        elif f in ("multisine", "chirp"):
            kwargs["center"] = center
        elif f == "sweep":
            # sweep() ramps to/from `start`, not a center to oscillate around -- reuse
            # `base` (this episode's actual starting pose: `inactive_value` when given,
            # e.g. range_sweep's jittered start pose, else the plain midpoint) so the
            # sweep begins and ends exactly where the episode's home-settle already put
            # the hand, matching how the held/inactive channels behave for the same
            # regime.
            kwargs["start"] = base
        fam_traj = _GENERATOR_FNS[f](duration_s, (lower, upper), family_seeds[f], **kwargs)
        traj[:, fam_channels] = fam_traj[:, fam_channels]

    traj = np.clip(traj, lower, upper)

    step_thresh = float(cl.load_latency()["step_thresh"])
    if max_delta is None:
        max_delta = float(rng.uniform(*ecfg.MAX_DELTA_RANGE))
    max_delta = min(max_delta, step_thresh)  # enforce the rate_limit() precondition
    traj = rate_limit(traj, max_delta, step_thresh)
    # rate_limit()'s continuous-integration path (out[t-1] + clamped delta) never
    # re-checks against [lower, upper] -- a channel driven toward an edge can walk past
    # it even though the pre-limiter traj was clipped (its own step-through path can't:
    # traj[t] there is already a clipped value). Real hardware violation measured before
    # this fix: up to 0.057 rad past the limit, only surfaced once CENTER_FRAC_RANGE/the
    # *_FRAC_RANGE floors were raised enough to push channels near the true edges.
    traj = np.clip(traj, lower, upper)

    delta = np.abs(np.diff(traj, axis=0)) if T > 1 else np.zeros((0, n))
    info = {
        "regime": regime,
        "families": list(families),
        "family_seeds": family_seeds,
        "active_channels": list(active_channels) if active_channels is not None else None,
        "steps_channels": steps_channels,
        "continuous_channels": continuous_channels,
        "continuous_family_channels": family_channels,
        "center": center.tolist(),
        "max_delta_rad": max_delta,
        "per_channel_delta_max": delta.max(axis=0).tolist() if delta.size else [0.0] * n,
        "per_channel_delta_p95": (np.percentile(delta, 95, axis=0).tolist()
                                   if delta.size else [0.0] * n),
    }
    return traj, step_events, info
