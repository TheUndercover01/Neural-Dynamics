# actuator_data — Shadow Hand Lite actuator-net data pipeline

Collects hardware data from a Shadow Hand Lite (right hand, ROS1 Noetic, PWM position
control) and turns it into actuator-network training samples. **The rosbag is the immutable
source of truth; all preprocessing is offline and re-runnable.**

## Sample definition
The input frame is built to be **identical to the RL policy's observation**
(`get_proprioception` in `my_policy_node_old.py`) so the actuator net drops into the policy's
obs space at inference.

- **Input X — 208 dims** = 4 stacked frames × 52. Each 52-feature frame is, in
  **`policy_joint_order`** (13):
  `[ pos_norm(13) | vel_norm(13) | err(13) | action(13) ]` where
  - `pos_norm`  = `process_value` normalised to [-1,1] by the joint's lower/upper limits
  - `vel_norm`  = `process_value_dot` normalised by ±vel limit
  - `err`       = controller `error`, **raw radians** (matches the policy)
  - `action`    = `set_point` normalised to [-1,1] by lower/upper limits
                  (for independent joints this equals the policy's raw action; at inference
                  you feed the policy output directly here)

  **Frame t is the LAST 52 elements**; oldest frame (t-3) is first. Limits and ordering live
  in `config/joints.yaml` under `policy_joint_order` / `limits`, copied from the policy node.
- **Output Y — 16 dims** = the 16 `/joint_states.position` values (the joints RViz draws),
  at `t + target_horizon` (default `+1`, a forward model).

**Coupled J0 is left "as is" (no ×2).** The three J0 actuators (`FFJ0/MFJ0/RFJ0`) drive a
coupled J1+J2 pair and the driver reports the *summed* value in `process_value`/`set_point`;
the coupled policy joint (`FFJ2/MFJ2/RFJ2`) is fed straight from that summed J0 signal. The
network learns the coupling from summed-actuator input → 16 individual joint outputs. Raw
radians are preserved in `data/aligned/`; the normalise+reorder into the policy frame happens
in `build_dataset.py`. See [DISCOVERY.md](DISCOVERY.md) for the live-verified topology.

## Layout
```
config/     joints.yaml (16 joints / 13 actuators / coupling), topics.yaml, pipeline.yaml
config_lib.py   single source of truth for canonical orders + the 52/208 frame layout
scripts/    discover.sh, record_episode.sh, check_stream.py, collect_dataset.sh (on-box)
preprocess/ parse_bag.py -> align.py -> build_dataset.py -> normalize.py   (offline)
qc/         report.py (per-episode PNG/HTML), loader_test.py (dataset contract test)
data/       raw/ (bags, source of truth, gitignored)  aligned/ (resampled npz, gitignored)
            dataset/ (X,Y + manifest, gitignored)
meta/       <session>/<episode>.json -- ONE JSON per episode: collection config
            (regime/families/seeds/max_delta/step_events/jitter/operator notes) +,
            once align.py runs, an "aligned" QC section. Git-tracked (unlike data/) so
            it transfers with a plain git pull.
```

## Quickstart
```bash
source /opt/ros/noetic/setup.bash
# 1. one-time: capture live topology/rates into DISCOVERY.generated.md
scripts/discover.sh
# 2. collect (see COLLECTION_PROTOCOL.md)
python3 scripts/check_stream.py
scripts/record_episode.sh 2026_07_02_am ep001 60
# 3. offline
python3 preprocess/align.py data/raw/2026_07_02_am/ep001_*.bag
python3 preprocess/build_dataset.py
python3 preprocess/normalize.py
python3 qc/loader_test.py
python3 qc/report.py data/aligned/2026_07_02_am/ep001_*.aligned.npz
```

## Key config knobs (`config/pipeline.yaml`)
| key | default | note |
|-----|---------|------|
| `dataset_rate` | 60 Hz | matches the deployed policy's control rate (hw + sim); also below both native rates |
| `stack_len` / `stack_stride` | 4 / 1 | frames per sample; 1 = consecutive |
| `target_horizon` | 1 | Y at t+1 (forward model); set 0 for same-step |
| `max_gap_ms` | 50 | grid points farther than this from data are dropped |
| `output_format` | hdf5 | auto-falls back to `.npz` if `h5py` missing |
| `split` | 0.7/0.15/0.15 | **by episode** (no temporal leakage) |

## Dependencies
- Collection: ROS1 Noetic (`rosbag`, `rospy`, `sensor_msgs`, `control_msgs`).
- Offline: `numpy`, `pyyaml` (present). Optional: `h5py` (HDF5 output — else npz),
  `matplotlib` (QC figures — else HTML tables only).

## Scope
Out of scope (provided elsewhere): the excitation/random-motion node, and the
actuator-net training code. This repo only produces `X/Y` datasets + `scaler.json`.
Open decisions and their current defaults are listed at the top of `pipeline.yaml`.
