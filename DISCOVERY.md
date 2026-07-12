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

## Tactile — /rh/tactile (sr_robot_msgs/BiotacAll) — VERIFIED LIVE 2026-07-11
The `shadow_touchlab_translator`/`tactile_relay` nodes `my_policy_node.py` reads tactile
through are **not running by default** (confirmed: no `/shadow_touchlab_translator/*`
topics present in `rostopic list`). We record the underlying hardware topic directly
instead — no extra nodes to launch, no dependency on those translators being up.

```
header: {seq, stamp, frame_id: "rh_distal"}
tactiles: [5]   # Biotac struct per finger, firmware order: ff, mf, rf, lf, th
  each Biotac: pac0, pac1, pac[20], pdc, tac, tdc, electrodes[19]
```
Live-sampled facts:
- `electrodes` carries **19** raw values per finger on the wire (the "16" only appears
  after `shadow_touchlab_translator`'s own truncation via its `n_taxels` launch param —
  we replicate that same truncation ourselves: `electrodes[:16]`).
- `tactiles[3]` (**lf**, little finger) reads **all-zero** live — confirmed, this hand
  has no little finger. Dropped entirely in our pipeline (not zero-padded).
- Real fingers ff/mf/rf/th (`tactiles[0,1,2,4]`) show live baseline readings in the
  ~700-1000 range (unloaded/at-rest).
- Publish rate: **~100 Hz** (measured directly via `rostopic hz /rh/tactile`, std dev
  0.0003s — a clean, steady rate, comfortably above the 60Hz dataset grid).

`preprocess/parse_bag.py` takes `electrodes[:16]` from each of the 4 real fingers
(indices 0,1,2,4) and concatenates them in that order (ff,mf,rf,th) into a flat 64-value
row per message → `gt_tactile` `[T,64]` after resampling. No summing/clustering/EMA/
baseline-subtraction is applied (that's `my_policy_node.py`'s own policy-observation
choice, out of scope here — this pipeline keeps full 64-taxel resolution).

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
| `gt_tactile` | `/rh/tactile` (`sr_robot_msgs/BiotacAll`), `electrodes[:16]` per real finger | no — raw, resampled only (finger selection/truncation matches `shadow_touchlab_translator`'s own convention, not a derived value) |
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
