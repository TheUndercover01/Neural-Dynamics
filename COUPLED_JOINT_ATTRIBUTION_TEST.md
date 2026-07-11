# Coupled joint (J1/J2) torque/velocity attribution test

Follow-on to `COUPLED_BACKLASH_TEST.md`. Script: `scripts/coupled_joint_attribution.py`.
Report: `outputs/coupled_joint_attribution.md`. Mapping (for sim use):
`outputs/coupled_joint_attribution_mapping.json`. Plots:
`outputs/coupled_joint_attribution_<finger>.png`. Raw data for the two new speed conditions:
`outputs/coupled_attribution_<finger>_speed<rad_s>.npz` (the slow condition reuses
`outputs/coupled_backlash_<finger>_raw.npz` from the earlier test). All raw data is saved so
this analysis can be re-run/re-tuned offline (`SKIP_LIVE=1 python3
scripts/coupled_joint_attribution.py`) without touching the hand again.

## Hypothesis

Each coupled finger has one motor and one force/velocity sensor driving a `[J1, J2]` pair.
User's question: since torque is applied to either J1 or J2 (not measured separately), does
the shared effort/velocity signal actually correlate with whichever joint's own angle is
changing at a given moment? If motion really is one-joint-at-a-time, a
`commanded_value -> active joint` one-hot map could let a sim/actuator-net decide which
joint's simulated torque to update from the one physical signal, and the shared signal could
plausibly be used to learn a per-joint torque model.

## Test

**Per-joint velocity is NOT available directly** — `/joint_states`' `velocity` field is, by
construction, identical for J1 and J2 (one shared motor sensor; verified in the earlier
mimic-joint velocity test). So each joint's own velocity was estimated by differentiating its
own POSITION signal (which IS sensed independently per joint): a centered difference over a
window sized to span a **fixed commanded-value range** (0.024 rad) rather than a fixed sample
count — see "Bug found and fixed" below for why that distinction matters.

**Three speed conditions**, reusing the safe hold pose and sweep mechanism from
`scripts/coupled_backlash_sweep.py` (imported directly, not duplicated):
- **slow** (0.1 rad/s) — reused the existing backlash-sweep data, no new hardware run.
- **medium** (0.3 rad/s) — new, 3 passes/finger.
- **fast** (0.8 rad/s) — new, 5 passes/finger.

Each sample classified into one of 4 states: `neither` / `J1-only` / `J2-only` / `both`, via
an activity threshold (0.02 rad/s) verified against each condition's own rest-state noise
floor (should stay well below threshold — confirmed, e.g. ~0.01-0.02 rad/s 95th percentile
vs the 0.02 threshold).

For each finger x speed x **direction** (extension/descending vs flexion/ascending,
separately — see Result), within J1-only and J2-only windows: Pearson correlation of
`effort` against that joint's own angle, and separately against that joint's own velocity.

## Bug found and fixed (before trusting any speed-trend claim)

Initial pass used a **fixed 15-sample** window for the centered difference at every speed.
`/joint_states` publishes at ~125Hz regardless of commanded sweep speed, so a fixed
sample-count window spans proportionally MORE commanded range as speed increases (verified:
0.024 rad at 0.1 rad/s -> 0.192 rad at 0.8 rad/s, an 8x change) — smearing the velocity
estimate across a sharp handoff and inflating the "both moving" fraction as a pure
resolution artifact, not a physical effect. Fixed by computing the window per-condition to
span a **fixed commanded-value range** instead (`window_for_speed()`); re-analysis (offline,
from the same saved raw data, no re-run needed) showed the speed-driven growth in "both"
shrank but did NOT disappear — e.g. RFJ0 flexion both-fraction: 0.5% (slow) -> 5.2%/2.2%
(medium, before/after fix) -> 52%/67-81% (fast, before/after fix). The growth-with-speed is
therefore a real effect, not purely the window artifact — but the artifact was real too and
is now corrected for.

## Result

**Direction matters more than initially assumed.** At the original slow (quasi-static) speed:

| Finger | Extension (descending) both-moving | Flexion (ascending) both-moving |
|---|---|---|
| FFJ0 | 44.8% | 0.5% |
| MFJ0 | 68.3% | 0.5% |
| RFJ0 | 3.8% | 0.5% |

Flexion (tendon being pulled taut — an unambiguous mechanical order) is essentially clean
one-hot for all three fingers. Extension (tendon slackening) shows real, substantial
simultaneous motion for FF/MF specifically, tracking the backlash magnitude already measured
in `COUPLED_BACKLASH_TEST.md` (MF > FF > RF).

**That picture does not hold at higher speed.** Both-moving grows substantially with speed in
BOTH directions:

| Direction | slow (0.1 rad/s) | fast (0.8 rad/s) |
|---|---|---|
| Extension | 39.0% (mean across fingers) | 77.3% |
| Flexion | 0.5% | 58.1% |

**Correlation (effort vs. the active joint's own signal), within single-joint-active
windows:** effort correlates more strongly, on average, with each joint's own **angle**
(mean \|r\|=0.71) than with its own **velocity** (mean \|r\|=0.44) — consistent with a
quasi-static, stiffness/tension-dominated torque regime rather than a velocity/damping-driven
one. Neither relationship is uniform: some cells are strong (e.g. RFJ0 extension, J2-angle
r=+0.99) while others are weak or slightly negative (e.g. J1-velocity during flexion, all
three fingers, all speeds).

## Verdict

A strict one-hot `commanded_value -> active joint` map is only defensible at the quasi-static
speeds actually tested here (~0.1 rad/s), and specifically during flexion. It should **not**
be assumed to hold at faster/more dynamic motion, or during extension, without checking data
at the actual speed used in sim. A single learned per-joint torque model from the shared
effort signal is similarly not a safe blanket assumption — angle is, on average, the stronger
predictor of effort within a joint's own active window, but the relationship's strength
varies meaningfully by finger, direction, and speed; the machine-readable per-condition,
per-direction mapping in `outputs/coupled_joint_attribution_mapping.json` is provided so a
sim consumer can pick the map matching its own operating speed/direction rather than using one
blended number.
