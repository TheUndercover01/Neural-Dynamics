# Coupled-joint tendon-slack / backlash test

Live-hardware investigation into whether tendon slack causes the coupled fingers'
second joint (J2) to "unlock" (start moving) earlier than expected during extension,
and whether that shows up as a torque anomaly. Script: `scripts/coupled_backlash_sweep.py`.
Report: `outputs/coupled_backlash_sweep.md`. Plots: `outputs/coupled_backlash_<finger>.png`.
Raw data: `outputs/coupled_backlash_<finger>_raw.npz` (every sample, so the analysis can be
re-run/re-tuned offline via `RAW_DIR=outputs python3 scripts/coupled_backlash_sweep.py`
without touching the hand again).

## Hypothesis

Each coupled finger (FF/MF/RF) has one motor driving a `[J1, J2]` pair through a shared
tendon (`config/joints.yaml` `coupling`). Nominally, extending a finger from full flexion
(180°) to straight (0°) should be a clean two-stage handoff: J1 moves alone from 180°→100°,
then J2 moves alone from 100°→0°. Tendon slack was suspected to make J2 start moving
*before* the descending command reaches the 100° handoff.

**User's hypothesis:** because a single motor drives both J1 and J2, the measured torque
should not meaningfully change depending on which of the two joints is currently the one
moving — i.e. an early/late unlock would be visible kinematically but *not* as a torque
signature.

## Command convention (verified live, not assumed)

The J0 controller's `/command` is the **raw summed J1+J2 angle**, range `0..π`
(`/sh_rh_ffj0_position_controller/command` etc.) — confirmed by commanding a value and
watching `process_value` settle to it directly. This is a *different* convention from
`excitation/run_simgap_hardware.py`'s `2× policy value` path (policy space is `0..π/2` per
joint); that path is unrelated to this test.

## Test

**Safety setup** (real hardware — done manually before scripting, then held by the script
for the duration):
- Thumb de-opposed (`THJ1=THJ2=THJ4=0`, `THJ5` swung to +1.0 rad) and visually confirmed
  clear of the index finger by the operator.
- FF/MF/RF spread apart via abduction (`FFJ4=-0.349`, `MFJ4=0.0`, `RFJ4=-0.349`) with knuckle
  `J3=0.65`, so a fully-flexed finger under test doesn't collide with its neighbors.
- Verified via each controller's `error` (`set_point − process_value`) staying ~0 — i.e. no
  joint mechanically blocked/straining — before and after every run.

**Sweep**, per coupled finger, one at a time (the other two + thumb held fixed at the above
pose throughout):
- Continuous (not step-and-hold) bidirectional ramp of the commanded sum: `π−0.05 ↔ 0.05`,
  at 0.1 rad/s — slow enough that friction/backlash dominates over dynamic/damping torque, a
  classic slow hysteresis-loop measurement.
- 3 full down+up passes per finger.
- Every arriving `/joint_states` message logged with its real timestamp and the commanded
  value in effect at that instant (no discrete grid, no settle-and-sample). ~22,800 samples
  logged per finger.
- Analysis interpolates each pass onto a common 120-point commanded-value grid and averages
  across the 3 passes.

**Unlock/active-region detection:** a joint is "active" where the smoothed slope
`|d(joint angle)/d(command)| > 0.3 rad/rad`, sustained over ≥10% of the grid (guards
against a single noisy point being mistaken for real motion), with the outer 3% of the grid
at each end excluded (that region is contaminated by a real but separate phenomenon — a
stiction/backlash "snap" transient right at each direction reversal, not the J1/J2 handoff).

## Result

| Finger | Measured J2 unlock | vs. 100° (1.745 rad) ideal | Backlash (hysteresis) J1 / J2 |
|---|---|---|---|
| FFJ0 | 2.35 rad (135°) | **EARLY by 35°** | 0.086 / 0.073 rad (4.9° / 4.2°) |
| MFJ0 | 2.73 rad (157°) | **EARLY by 57°** | 0.154 / 0.137 rad (8.8° / 7.8°) |
| RFJ0 | 1.66 rad (95°) | LATE by 5° (≈matches) | 0.023 / 0.019 rad (1.3° / 1.1°) |

("Early" = J2 measurably active while the descending command is still above the 100° ideal
handoff; "late" = J1 keeps moving past it, J2 stays locked longer.)

**Torque:** for all three fingers, `|d(effort)/d(command)|` near each finger's *own* measured
unlock point was **less steep** than elsewhere on the sweep (ratios 0.13×, 0.14×, 0.76× — all
< 1). No distinct torque spike at the handoff for any finger.

**Verdict: hypothesis supported.** FF and MF show a real, sizeable early unlock (35–57°,
consistent with tendon slack) and RF is close to the ideal reference — but none of the three
show a corresponding torque anomaly at their own unlock point. This is consistent with one
motor/one tendon/one force sensor driving the pair: the sensor cannot distinguish which of
the two joints is currently yielding, so torque tracks the commanded value smoothly
regardless of the kinematic handoff.

## Caveat / how to re-tune

"Unlock" is threshold-defined (0.3 rad/rad, sustained ≥10% of the grid). For FF/MF this
lands where J2's motion has *mostly* tapered off, not necessarily the sharpest visual kink
in the position-hysteresis plot (which looks closer to ~60-65° for FFJ0). Raising
`ACTIVE_SLOPE_THRESH` (env var) and re-running the analysis against the saved raw `.npz`
files (no hardware needed) will shift toward detecting that sharper kink instead.

## First attempt was buggy — corrected before trusting these numbers

An earlier pass at this analysis reported the *identical* unlock value (down to 3 decimal
places) for all three physically-independent fingers — the tell that it was a code bug, not
physics. Root cause: `j2_unlock_cmd` was computed as `grid[j2_active].max()` (global max over
a scattered boolean mask) instead of the boundary of the *contiguous* active region, so a
single-point stiction/reversal-transient spike at the sweep's edge corrupted the result to
the grid's boundary value for every finger. Fixed via `_smooth()` + `_find_active_edge()`
(contiguous-run detection with a sustained-inactive requirement) plus explicit boundary
trimming, and validated against synthetic data shaped like the real signal before re-running
on hardware a second time.
