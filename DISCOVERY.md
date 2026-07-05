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
- `error`             — `set_point - process_value` → **act_err** feature
- `command`           — controller output (**PWM** in this mode) → context only, not in 208

Publish rate: **~87 Hz** (prior live session).

Each actuator also exposes `.../command`, `.../max_force_factor`, `.../pid/*`, and the J0
actuators additionally publish `.../underactuation_cartesian_error` (not used).

## Rate note
`rostopic hz` and subscribing to the controller `/state` topics could not be completed
from the current container: the `/joint_states` publisher is reachable (a single
`rostopic echo -n1` succeeded) but subscriptions to the controller-state publishers hang
(publisher advertises a host URI this container cannot route to). **Re-run
`scripts/discover.sh` from a node on the hand's own network** to fill in measured rates.
Until then configs use the prior-session values (125 / 87 Hz) and `dataset_rate: 100`.
