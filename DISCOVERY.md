# DISCOVERY — Shadow Hand Lite (rh_), ROS1 Noetic

Captured live from the running hand on **2026-07-02**. This file is the ground-truth
record of topology/rates the configs were built from. Re-run `scripts/discover.sh` on
the box to refresh it (that script writes machine-captured sections; the notes below are
the human-verified summary).

## Environment
- ROS distro: **noetic** (ROS1). Source `/opt/ros/noetic/setup.bash` before anything.
- Hand: **right** (`rh_` prefix, uppercase joint names). A stray `/lh_trajectory_controller`
  topic exists in the graph but is unrelated to this hand — ignore it.
- Control mode: **PWM position control**. Confirmed by PID params exposing
  `.../pid/max_pwm`, `.../pid/deadband`, and strain-gauge refs (`sg_left`, `sg_right`,
  `sgleftref`) under `/rh/<joint>/pid/*`. The controller `command` field is therefore PWM.

## /joint_states  (sensor_msgs/JointState)
16 joints, FIXED order (index = array position, verified live):

```
0  rh_FFJ1   1  rh_FFJ2   2  rh_FFJ3   3  rh_FFJ4
4  rh_MFJ1   5  rh_MFJ2   6  rh_MFJ3   7  rh_MFJ4
8  rh_RFJ1   9  rh_RFJ2  10  rh_RFJ3  11  rh_RFJ4
12 rh_THJ1  13  rh_THJ2  14  rh_THJ4  15  rh_THJ5
```

Aligned arrays: `name[i]`, `position[i]`, `velocity[i]`, `effort[i]`.
Publish rate: **~125 Hz** (from prior live session; not re-measured — see note).

### J0 coupling — CONFIRMED FROM LIVE DATA
In a single `/joint_states` sample the J1/J2 pairs of the three coupled fingers share an
identical velocity and identical effort (one motor drives the tendon pair) but report
different positions (two independent joint sensors):

| pair        | velocity (both)        | effort (both) |
|-------------|------------------------|---------------|
| FFJ1 = FFJ2 | 0.00022405161126778353 | 4.396068…     |
| MFJ1 = MFJ2 | -0.0005870898586764557 | 17.31948…     |
| RFJ1 = RFJ2 | 0.00045509459544198014 | 12.75682…     |

This is the J0 signature: `FFJ0 = FFJ1 + FFJ2`, etc.

## Controller state topics  (control_msgs/JointControllerState)
Exactly **13** `/state` topics, one per actuator — matches `actuator_order` 1:1:

```
/sh_rh_ffj0_position_controller/state   /sh_rh_ffj3_position_controller/state
/sh_rh_ffj4_position_controller/state   /sh_rh_mfj0_position_controller/state
/sh_rh_mfj3_position_controller/state   /sh_rh_mfj4_position_controller/state
/sh_rh_rfj0_position_controller/state   /sh_rh_rfj3_position_controller/state
/sh_rh_rfj4_position_controller/state   /sh_rh_thj1_position_controller/state
/sh_rh_thj2_position_controller/state   /sh_rh_thj4_position_controller/state
/sh_rh_thj5_position_controller/state
```

Fields used (per JointControllerState):
- `set_point`         — commanded target  → **action** feature
- `process_value`     — measured joint position (for J0 = summed J1+J2) → **act_pos** feature
- `process_value_dot` — measured velocity → **act_vel** feature
- `error`             — `process_value - set_point` (verified live 2026-07-11: matches this
  sign, not the reverse; see actuator_data's align.py output) → **act_err** feature
- `command`           — controller output (**PWM** in this mode) → context only, not in 208

Publish rate: **~87 Hz** (prior live session).

Each actuator also exposes `.../command`, `.../max_force_factor`, `.../pid/*`, and the J0
actuators additionally publish `.../underactuation_cartesian_error` (not used).

## Tactile — /shadow_touchlab_translator/calibrated — VERIFIED LIVE 2026-07-12
**Correction from an earlier version of this section**: we first recorded the raw
hardware topic `/rh/tactile` directly to avoid depending on extra nodes. That gives raw
ADC electrode counts, NOT the calibrated signal `my_policy_node.py` actually uses (it
reads `/shadow_touchlab_translator/calibrated_flat`, downstream of the `calibrated`
topic here). We switched to recording the real calibrated topic instead, which requires
the `shadow_touchlab_translator` node running — `record_episode.sh` now auto-launches
it (`roslaunch touchlab_driver_ros translator.launch`) if not already up, idempotently
(a multi-episode session only pays the launch cost once).

The underlying hardware topic is still `/rh/tactile` (`sr_robot_msgs/BiotacAll`,
`tactiles[5]`, firmware order ff/mf/rf/lf/th, `Biotac.electrodes[19]` raw values/finger,
`lf` confirmed all-zero live — no little finger on this hand). The translator node
subscribes that, takes `electrodes[:16]` per finger (`n_taxels=16` launch param) x10.0
scale, then runs it through `touchlab_comm_py`'s calibration (`self.com.translate()`,
using `/ros1/calibration/uoe-default.bin`) and auto-zeros to the first reading it sees
after starting. It publishes two topics (both `touchlab_msgs/Float64MultiArrayStamped`):
- `.../raw` — 80 values (5 fingers x 16 taxels, the same x10-scaled electrode data).
- `.../calibrated` — **240 values**, confirmed live: only 4 were non-zero at rest, and
  all 4 sat at index ≡2 (mod 3) within their triplet. So this is 80 taxel positions x 3
  components each, and only the 3rd component per taxel (`data[2::3]`) is meaningfully
  non-zero at rest (components 0/1 read ~0 untouched — likely shear vs. normal force).
  This confirms the `raw[2::3]` convention already used by `my_policy_node.py`/
  `run_simgap_hardware.py` for their own (differently-sourced) tactile topic.
- Publish rate: **~100 Hz** (measured directly via `rostopic hz`, though jitter is
  higher than the raw hardware topic — std dev ~0.015s vs. 0.0003s — from the
  translator's own per-message calibration compute).

`preprocess/parse_bag.py` unpacks `calibrated`'s 240 values via `[2::3]` to 80 per-taxel
values (firmware order ff/mf/rf/lf/th), then keeps the 4 REAL fingers (ff,mf,rf,th; `lf`
slice `[48:64]` dropped entirely, not zero-padded) → a flat 64-value row per message →
`gt_tactile [T,64]` after resampling. No summing/clustering/EMA/baseline-subtraction on
top of that — this pipeline keeps full 64-taxel resolution (that further reduction is
`my_policy_node.py`'s own policy-observation choice, out of scope here).

## What's raw hardware data vs. computed by our pipeline
Every field listed above (`/joint_states.position/velocity/effort`, and the controller's
`set_point`/`process_value`/`process_value_dot`/`error`/`command`) is **raw, exactly as
the driver/controller firmware publishes it** — no script in this repo derives or
recalculates any of these values. `preprocess/align.py` only *resamples* them onto the
common 60Hz grid (`zoh`: holds the last real sample; never blends, averages, or derives a
new value):

| aligned column | raw source field | computed by us? |
|---|---|---|
| `gt_pos` | `/joint_states.position` | no — raw, resampled only |
| `gt_vel` | `/joint_states.velocity` | no — raw, resampled only |
| `gt_effort` | `/joint_states.effort` | no — raw, resampled only |
| `gt_tactile` | `/shadow_touchlab_translator/calibrated`, `[2::3]`-unpacked per real finger | no — raw passthrough of the translator node's own calibrated output, resampled only (the `[2::3]` unpack recovers the meaningful component per taxel from the message's own 3-per-taxel packing; the calibration itself is computed by `shadow_touchlab_translator`/`touchlab_comm_py`, not by this repo) |
| `action` | controller `set_point` | no — raw, resampled only |
| `act_pos` | controller `process_value` | no — raw, resampled only |
| `act_vel` | controller `process_value_dot` | no — raw, resampled only |
| `act_err` | controller `error` (sign verified live: `process_value - set_point`) | no — raw, resampled only |
| `command` | controller `command` (PWM) | no — raw, resampled only |

These, by contrast, ARE computed/derived by this repo's own code, not read directly off
any topic:
- **`t`, `valid`, `seg_id`** (`preprocess/align.py`) — the uniform time grid and
  gap/segment bookkeeping; synthetic, not a sensor reading.
- **`max_delta_rad`, `per_channel_delta_max`/`_p95`, `step_events`**
  (`excitation/compose.py`) — properties of the trajectory WE generated and commanded,
  logged as ground truth for what was intentionally excited; not derived from any
  sensor reading.
- **`home_pose_actuator`** (`excitation/config.py`) — the baoding-ball pose converted
  from policy-space into actuator-space radians (coupled joints x2), clipped to
  `effective_command_limits()`. This is what we command the hand to; it is not a sensor
  reading.
- **`pre_episode_setpoint_gap`** (`excitation/publisher.py`) — `|process_value -
  set_point|`, computed by us from two raw fields right before each episode's
  trajectory starts.

## Rate note
`rostopic hz` and subscribing to the controller `/state` topics could not be completed
from the current container: the `/joint_states` publisher is reachable (a single
`rostopic echo -n1` succeeded) but subscriptions to the controller-state publishers hang
(publisher advertises a host URI this container cannot route to). **Re-run
`scripts/discover.sh` from a node on the hand's own network** to fill in measured rates.
Until then configs use the prior-session values (125 / 87 Hz) for the native topic rates.
`dataset_rate: 60` is fixed independently of those — it matches the deployed policy's
control rate (hw + sim), not a value derived from the native topic rates.
