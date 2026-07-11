"""Pure-Python trajectory generators. No ROS imports anywhere in this module — every
function is offline-testable (see test_generators.py).

Each generator has the shape (duration_s, joint_limits, rng_seed, **params) -> (T,13)
float64 radians, in actuator_order, T = round(duration_s * rate). `joint_limits` is
(lower[13], upper[13]) — pass excitation.config.effective_command_limits() for the
inset-from-hard-stops range used at collection time.

`channels` (default: all 13) restricts which columns are dynamic; the rest are held at
a constant midpoint. compose.py uses this to assign disjoint actuators to different
families (e.g. `steps` owns a few channels, `ou_walk` owns the rest) so a step's
post-jump hold band is never disturbed by an overlaid continuous signal — see
compose.py's module docstring for why that matters.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))         # same dir: config.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))     # repo root: config_lib.py
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402


def _unpack_limits(joint_limits):
    lower, upper = joint_limits
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return lower, upper


def _resolve_channels(channels, n):
    return list(range(n)) if channels is None else list(channels)


def _step_thresh() -> float:
    return float(cl.load_latency()["step_thresh"])


# =========================================================================================
# ou_walk — per-actuator Ornstein-Uhlenbeck (mean-reverting random walk)
# =========================================================================================

def ou_walk(duration_s, joint_limits, rng_seed, *, theta=None, sigma_frac=None, mu=None,
            rate=None, channels=None):
    """Discrete OU process per channel: x += theta*(mu-x)*dt + sigma*sqrt(dt)*N(0,1).

    theta (mean-reversion rate, 1/s) and sigma_frac (noise std, fraction of joint
    half-range) are randomized per-channel within excitation.config bounds when not
    given, so a single episode mixes slow drift (low theta) and jitter (high theta).
    Per-step |dx| is hard-clipped to CONTINUOUS_SLEW_FRAC * step_thresh regardless of
    theta/sigma — see excitation/config.py's CONTINUOUS_SLEW_FRAC comment for why.
    """
    rate = rate or ecfg.PUBLISH_RATE_HZ
    dt = 1.0 / rate
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)
    channels = _resolve_channels(channels, n)

    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    anim = np.zeros(n, dtype=bool)
    anim[channels] = True

    theta_arr = (rng.uniform(*ecfg.OU_THETA_RANGE, size=n) if theta is None
                 else np.broadcast_to(theta, (n,)).astype(float))
    sigma_frac_arr = (rng.uniform(*ecfg.OU_SIGMA_FRAC_RANGE, size=n) if sigma_frac is None
                       else np.broadcast_to(sigma_frac, (n,)).astype(float))
    sigma_arr = np.where(anim, sigma_frac_arr * half_range, 0.0)
    theta_arr = np.where(anim, theta_arr, 0.0)
    mu_arr = midpoint.copy() if mu is None else np.broadcast_to(mu, (n,)).astype(float).copy()

    max_delta = ecfg.CONTINUOUS_SLEW_FRAC * _step_thresh()
    sqrt_dt = np.sqrt(dt)

    traj = np.empty((T, n), dtype=float)
    x = mu_arr.copy()
    for t in range(T):
        traj[t] = x
        noise = rng.standard_normal(n) * sigma_arr * sqrt_dt
        dx = theta_arr * (mu_arr - x) * dt + noise
        dx = np.clip(dx, -max_delta, max_delta)
        x = x + dx
    return np.clip(traj, lower, upper)


# =========================================================================================
# multisine — sum of incommensurate sinusoids per joint
# =========================================================================================

def multisine(duration_s, joint_limits, rng_seed, *, n_components=None, freq_range=None,
              amp_frac=None, rate=None, channels=None):
    """Sum of 3-6 (config-bounded) sines per channel, random freq/phase/amplitude split.

    Each component's amplitude is capped so amp * 2*pi*freq (its max slope) cannot push
    the per-frame delta past its share of CONTINUOUS_SLEW_FRAC * step_thresh — otherwise
    a handful of fast, large-amplitude components could sum to a slope that trips
    detect_steps(), silently turning a "continuous" family into false step events.
    """
    rate = rate or ecfg.PUBLISH_RATE_HZ
    dt = 1.0 / rate
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)
    channels = _resolve_channels(channels, n)
    freq_range = freq_range or ecfg.MULTISINE_FREQ_HZ_RANGE

    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    t = np.arange(T) / rate

    max_total_delta = ecfg.CONTINUOUS_SLEW_FRAC * _step_thresh()

    traj = np.broadcast_to(midpoint, (T, n)).copy()
    for c in channels:
        k = (n_components if n_components is not None
             else int(rng.integers(ecfg.MULTISINE_N_COMPONENTS_RANGE[0],
                                    ecfg.MULTISINE_N_COMPONENTS_RANGE[1] + 1)))
        amp_total_frac = amp_frac if amp_frac is not None else rng.uniform(*ecfg.MULTISINE_AMP_FRAC_RANGE)
        weights = rng.dirichlet(np.ones(k))
        freqs = rng.uniform(freq_range[0], freq_range[1], size=k)
        phases = rng.uniform(0.0, 2 * np.pi, size=k)

        budget_per_component = max_total_delta / k
        signal = np.zeros(T)
        for j in range(k):
            desired_amp = weights[j] * amp_total_frac * half_range[c]
            slew_cap_amp = budget_per_component / max(2 * np.pi * freqs[j] * dt, 1e-9)
            amp_j = min(desired_amp, slew_cap_amp)
            signal += amp_j * np.sin(2 * np.pi * freqs[j] * t + phases[j])
        traj[:, c] = midpoint[c] + signal
    return np.clip(traj, lower, upper)


# =========================================================================================
# chirp — frequency sweep per joint, staggered phase across joints
# =========================================================================================

def chirp(duration_s, joint_limits, rng_seed, *, freq_range=None, amp_frac=None,
          sweep="linear", rate=None, channels=None):
    """Linear or log frequency sweep f0->f1 per channel. Each channel gets an
    independent random start phase (`phase0`) so joints don't sweep in lockstep.
    Amplitude is capped against the sweep's max instantaneous slope, same rationale
    as multisine's per-component cap.
    """
    rate = rate or ecfg.PUBLISH_RATE_HZ
    dt = 1.0 / rate
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)
    channels = _resolve_channels(channels, n)
    freq_range = freq_range or ecfg.CHIRP_FREQ_HZ_RANGE
    f0, f1 = freq_range
    duration = T / rate

    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    t = np.arange(T) / rate

    max_delta = ecfg.CONTINUOUS_SLEW_FRAC * _step_thresh()
    f_peak = max(f0, f1, 1e-9)

    traj = np.broadcast_to(midpoint, (T, n)).copy()
    for c in channels:
        desired_amp = (amp_frac if amp_frac is not None
                        else rng.uniform(*ecfg.CHIRP_AMP_FRAC_RANGE)) * half_range[c]
        slew_cap_amp = max_delta / (2 * np.pi * f_peak * dt)
        amp = min(desired_amp, slew_cap_amp)
        phase0 = rng.uniform(0.0, 2 * np.pi)  # staggered start phase per joint

        if sweep == "log" and f0 > 0 and f1 > 0 and f0 != f1:
            k = (f1 / f0) ** (1.0 / duration)
            phase = 2 * np.pi * f0 * (k ** t - 1.0) / np.log(k)
        else:
            phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / duration)
        traj[:, c] = midpoint[c] + amp * np.sin(phase + phase0)
    return np.clip(traj, lower, upper)


# =========================================================================================
# steps — hold -> jump -> hold ~1s, randomized target + magnitude
# =========================================================================================

def _step_segments(duration_s, joint_limits, rng_seed, channels, rate,
                    hold_s_range, magnitude_frac_range):
    """Shared by steps() and step_schedule() so both agree exactly: per channel, a list
    of (start_frame, end_frame_exclusive, held_value) segments."""
    lower, upper = _unpack_limits(joint_limits)
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)
    hold_s_range = hold_s_range or ecfg.STEP_HOLD_S_RANGE
    magnitude_frac_range = magnitude_frac_range or ecfg.STEP_MAGNITUDE_FRAC_RANGE
    latency_cfg = cl.load_latency()
    step_thresh = float(latency_cfg["step_thresh"])
    # detect_steps() (diagnostics/latency_timeconstant.py) rejects a jump unless the
    # command then holds within a tight band for >= hold_frames afterward. A step placed
    # too close to the trajectory's end can't satisfy that, so it must not be scheduled —
    # otherwise step_schedule() would list a "step" the real detector silently drops.
    min_verifiable_hold = int(latency_cfg["hold_frames"])

    segments: dict[int, list[tuple[int, int, float]]] = {}
    for c in channels:
        span = upper[c] - lower[c]
        min_mag = max(step_thresh * 1.5, magnitude_frac_range[0] * span)
        max_mag = max(min_mag + 1e-6, magnitude_frac_range[1] * span)
        segs: list[tuple[int, int, float]] = []
        frame = 0
        value = lower[c] + rng.uniform(0.3, 0.7) * span
        while frame < T:
            hold_frames = int(round(rng.uniform(*hold_s_range) * rate))
            end = min(frame + max(hold_frames, 1), T)
            segs.append((frame, end, value))
            frame = end
            if frame >= T:
                break
            remaining = T - frame
            # detect_steps rejects when (jump_frame + hold_frames) >= T, i.e. needs
            # remaining STRICTLY > hold_frames — remaining == hold_frames still fails.
            if remaining <= min_verifiable_hold:
                # Not enough room left for a verifiable post-jump hold: extend this hold
                # to the end instead of emitting a step detect_steps would reject anyway.
                s, e, v = segs[-1]
                segs[-1] = (s, T, v)
                break
            candidate = value
            for _ in range(20):
                mag = rng.uniform(min_mag, max_mag)
                direction = rng.choice([-1.0, 1.0])
                trial = value + direction * mag
                if lower[c] <= trial <= upper[c]:
                    candidate = trial
                    break
            else:
                candidate = lower[c] if value > (lower[c] + upper[c]) / 2 else upper[c]
            value = float(np.clip(candidate, lower[c], upper[c]))
        segments[c] = segs
    return segments, T


def steps(duration_s, joint_limits, rng_seed, *, rate=None, channels=None,
          hold_s_range=None, magnitude_frac_range=None):
    """Piecewise-constant per channel: hold, jump by > step_thresh, hold ~1s, repeat."""
    rate = rate or ecfg.PUBLISH_RATE_HZ
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    channels = _resolve_channels(channels, n)
    midpoint = 0.5 * (lower + upper)
    segments, T = _step_segments(duration_s, joint_limits, rng_seed, channels, rate,
                                  hold_s_range, magnitude_frac_range)
    traj = np.broadcast_to(midpoint, (T, n)).copy()
    for c, segs in segments.items():
        for start, end, value in segs:
            traj[start:end, c] = value
    return np.clip(traj, lower, upper)


def step_schedule(duration_s, joint_limits, rng_seed, *, rate=None, channels=None,
                   hold_s_range=None, magnitude_frac_range=None):
    """Exact step events matching steps()'s jumps: list of dicts with frame, actuator_idx
    (index into actuator_order), pre, post, magnitude. Ground truth for the meta/ JSON sidecar and for
    the offline test that checks detect_steps() recovers exactly these events."""
    rate = rate or ecfg.PUBLISH_RATE_HZ
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    channels = _resolve_channels(channels, n)
    segments, _T = _step_segments(duration_s, joint_limits, rng_seed, channels, rate,
                                   hold_s_range, magnitude_frac_range)
    events = []
    for c, segs in segments.items():
        for i in range(1, len(segs)):
            pre = segs[i - 1][2]
            post = segs[i][2]
            frame = segs[i][0]
            events.append({"frame": frame, "actuator_idx": c, "pre": float(pre),
                            "post": float(post), "magnitude": float(post - pre)})
    events.sort(key=lambda e: e["frame"])
    return events


# =========================================================================================
# static_holds — fixed targets across the range, near-zero velocity, randomized dwell
# =========================================================================================

def static_holds(duration_s, joint_limits, rng_seed, *, rate=None, channels=None,
                  dwell_s_range=None):
    """Per channel: sit at a random target for a random dwell, then move to the next
    random target via a slow ramp (never a jump) — the stiction/deadband family.

    The inter-target ramp length is sized from the transition's own magnitude so its
    per-frame delta never approaches step_thresh: this family must never register as a
    step, only as a settled hold.
    """
    rate = rate or ecfg.PUBLISH_RATE_HZ
    lower, upper = _unpack_limits(joint_limits)
    n = lower.size
    T = int(round(duration_s * rate))
    rng = np.random.default_rng(rng_seed)
    channels = _resolve_channels(channels, n)
    dwell_s_range = dwell_s_range or ecfg.STATIC_HOLD_DWELL_S_RANGE
    max_ramp_delta = ecfg.CONTINUOUS_SLEW_FRAC * _step_thresh()
    midpoint = 0.5 * (lower + upper)

    traj = np.broadcast_to(midpoint, (T, n)).copy()
    for c in channels:
        frame = 0
        value = float(rng.uniform(lower[c], upper[c]))
        while frame < T:
            dwell_frames = int(round(rng.uniform(*dwell_s_range) * rate))
            end = min(frame + max(dwell_frames, 1), T)
            traj[frame:end, c] = value
            frame = end
            if frame >= T:
                break
            new_value = float(rng.uniform(lower[c], upper[c]))
            ramp_frames = max(1, int(np.ceil(abs(new_value - value) / max_ramp_delta)))
            # Build the FULL-length ramp (so its per-frame delta respects max_ramp_delta),
            # then only write the prefix that fits before T. Truncating ramp_frames itself
            # instead would compress the same total distance into fewer frames and spike
            # the per-frame delta past the cap right at the trajectory's tail.
            full_ramp = np.linspace(value, new_value, ramp_frames, endpoint=False)
            write_end = min(frame + ramp_frames, T)
            write_len = write_end - frame
            traj[frame:write_end, c] = full_ramp[:write_len]
            frame = write_end
            value = new_value
    return np.clip(traj, lower, upper)
