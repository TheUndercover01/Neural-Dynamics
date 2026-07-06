# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration mode: mentor, not replacement

You are my programming mentor, not my replacement. Rules:

1. Never implement a feature before presenting a plan.
2. Never modify more than 2-3 files at once unless I explicitly ask.
3. After every major function, explain:
   - why it exists
   - how it works
   - alternative implementations
4. Ask me architecture questions before coding.
5. Prefer guiding me over immediately giving the full solution.
6. Keep solutions minimal. Don't introduce abstractions unless necessary.
7. Stop after each milestone and wait for my review.

## What this repo is

A ROS1 Noetic data-collection + offline-preprocessing pipeline for training an actuator
network for a Shadow Hand Lite (right hand, `rh_` prefix, PWM position control). It turns
rosbags into `(X, Y)` training samples shaped to match a specific pre-trained RL policy's
observation space. **The rosbag is the immutable source of truth** — every derived artifact
under `data/aligned/` and `data/dataset/` is disposable and re-runnable from `data/raw/*.bag`.
If you change a config, delete and regenerate downstream data rather than patching it.

The excitation/motion-generation node and the actuator-net training code both live outside
this repo; this repo's job stops at producing datasets + `scaler.json`.

## Commands

Everything downstream of collection is plain `numpy`/`pyyaml` — no ROS install needed.
`rosbag`/`rospy` imports in `preprocess/parse_bag.py` and `scripts/check_stream.py` are
lazy (inside functions, not at module top), specifically so the rest of the pipeline runs
on a machine with no ROS. `h5py` and `matplotlib` are optional (silent `.npz` / HTML-only
fallback if missing — see `build_dataset.py:_write` and `qc/report.py`).

```bash
source /opt/ros/noetic/setup.bash        # only needed for collection-stage scripts

# --- Collection (on the robot box, live ROS master) ---
scripts/discover.sh                                    # -> DISCOVERY.generated.md
python3 scripts/check_stream.py                        # preflight; DURATION=<sec> env override
scripts/record_episode.sh SESSION_ID EPISODE_ID [SEC]  # -> data/raw/<session>/*.bag + meta.yaml
                                                         # env: EXCITATION=, CONTROL_MODE=, WARMUP=, DRIVER_VERSION=, OPERATOR=

# --- Offline preprocessing (no ROS required) ---
python3 preprocess/parse_bag.py data/raw/<session>/*.bag        # prints rate/drop/jitter QC, no output file
python3 preprocess/align.py data/raw/<session>/<ep>.bag         # -> data/aligned/<session>/<ep>.aligned.npz(+.json)
python3 preprocess/build_dataset.py [ALIGNED_NPZ ...]           # -> data/dataset/<session>/<ep>.h5(or .npz) + manifest.json
python3 preprocess/normalize.py                                 # -> data/dataset/scaler.json (train-episode stats only)

# --- QC / the closest thing to a test suite ---
python3 qc/loader_test.py [FILE ...]      # shape/NaN/frame-order contract check; run after
                                           # touching build_dataset.py or config_lib.py
python3 qc/report.py data/aligned/<session>/<ep>.aligned.npz    # per-episode plots + HTML
```

There is no pytest suite. `qc/loader_test.py` is the correctness gate — run it standalone
against one file (`python3 qc/loader_test.py data/dataset/<session>/<ep>.h5`) or with no
args to check every episode in `data/dataset/manifest.json`.

## Architecture

### `config_lib.py` is the single source of truth
Every script does the same `sys.path.insert(...)` + `import config_lib as cl` rather than
hardcoding joint names, topic names, or frame geometry. It loads and validates two YAML
files under `config/`:

- **`joints.yaml`** — `joint_order` (16, must match `/joint_states` wire order exactly),
  `actuator_order` (13, physical controller order), `coupling` (J0 → its `[J1,J2]` joint
  pair), plus the **policy-frame** section: `policy_joint_order`, `policy_to_actuator`,
  `limits`. `config_lib.load_joints()` asserts the 16/13 lengths at load time, so an edit
  that breaks the count fails fast on the next import, everywhere.
- **`pipeline.yaml`** — `dataset_rate`, per-signal `interp` mode (`zoh` for commands,
  `linear` for measurements — see the comment block in the file), `stack_len`/`stack_stride`,
  `target_horizon`, episode split ratios, `output_format`.
- **`topics.yaml`** — exact topic strings and expected rates.

### Two coordinate systems — don't conflate them
1. **Actuator order** (13, `actuator_order`) — physical driver order: `FFJ0, FFJ3, FFJ4,
   MFJ0, ...`. This is how `align.py` stores columns in `data/aligned/*.npz`
   (`act_pos`/`act_err`/`act_vel`/`action`).
2. **Policy order** (13, `policy_joint_order`) — the order and normalization the trained RL
   policy's observation uses (mirrors `get_proprioception()` in the external policy node,
   not part of this repo). `build_dataset.py` converts actuator-order → policy-order via
   `config_lib.policy_perm()`, then normalizes pos/vel/action to `[-1,1]` via
   `config_lib.policy_limits()` (`err` is left in raw radians, matching the policy). This
   conversion is what makes the produced `X` directly usable as actuator-net input shaped
   like the policy's own observation.

The three coupled fingers (FF/MF/RF) have one motor driving two joints (J1+J2); the J0
actuator already reports the **summed** position/set_point. The policy frame's coupled slot
(e.g. `rh_FFJ2`) is fed straight from the summed `rh_FFJ0` value with **no rescaling** — the
network is expected to learn the coupling itself. `Y` is always the 16 individual joints
from `/joint_states`.

### Pipeline stages and where boundaries matter
`parse_bag.py` (bag → raw per-topic arrays) → `align.py` (resample every stream onto one
time grid at `dataset_rate`; per-signal `zoh`/`linear` per `pipeline.yaml`; grid points
farther than `max_gap_ms` from real data are marked invalid; `align_parsed()` assigns
`seg_id`, a contiguous-valid-run id, `-1` where invalid) → `build_dataset.py` (stacks
`stack_len` consecutive frames into `X`, looks `target_horizon` steps ahead for `Y`; a
stacked window is **dropped** if it crosses a `seg_id` boundary — see the loop in
`build_one()` — since stacking must never span a data gap or an episode file) →
`normalize.py` (mean/std over the **train**-split episodes only; the split is a seeded
permutation over whole episodes, never over individual frames, to avoid temporal leakage).

### Frame layout — the "208" and "52"
`X` is `stack_len` (4) frames of `FRAME_DIM` (52) each; **frame `t` occupies the LAST 52
columns** of `X`, oldest frame first. Each 52-frame is
`[pos_norm(13) | vel_norm(13) | err(13) | action(13)]` in `policy_joint_order` — the group
order/count is defined once as `config_lib.FRAME_FEATURE_GROUPS` and must not be reordered
without regenerating every downstream dataset (`manifest.json` records `input_columns` so
you can always recover the exact layout used). `Y` is the 16 raw `/joint_states` positions
in `joint_order`, `target_horizon` grid-steps after the input window's last frame.

### Live-hardware caveats (see `DISCOVERY.md`)
Topology facts in `joints.yaml`/`topics.yaml` were verified against a live hand, but
`rostopic hz` and subscribing to the 13 controller `/state` topics could not be completed
from every container — the publisher may advertise a host URI unreachable from a given
sandbox, even though `/joint_states` itself is reachable. Re-run `scripts/discover.sh` from
a node on the hand's own network before trusting rate assumptions on a new box.
