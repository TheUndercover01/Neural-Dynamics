# Collection Protocol — Shadow Hand Lite actuator-net data

The excitation / random-motion source is **out of scope here** — this pipeline logs whatever
motion is commanded (it reads `set_point` from the controller side, so any motion node works).
This document is the operating procedure for producing clean, immutable episode bags.

## Before anything
```bash
source /opt/ros/noetic/setup.bash
```

## 0. Warm-up (once per session)
Tendon-driven joints stiffen when cold. **Run motion for ~2 minutes before the first
recorded episode** so tendon tension/temperature settle. Mark the first post-warm-up
episodes as `WARMUP=warm`; if you deliberately capture a cold run, tag it `WARMUP=cold`.

## 1. Pre-flight (every session, and any time a controller looks off)
```bash
python3 scripts/check_stream.py        # DURATION=5 default
```
Must print `RESULT: PASS`. It verifies:
- all 14 topics (`/joint_states` + 13 controller states) are publishing;
- per-actuator `set_point` vs `process_value` look alive (not frozen/limp);
- the J0 coupling holds: `FFJ0.process_value ≈ FFJ1+FFJ2` (wiring/calibration sanity).

If a controller is `SILENT` or the coupling `MISMATCH`es, stop and fix before recording —
don't waste a run.

## 2. Record episodes
```bash
# record_episode.sh SESSION_ID EPISODE_ID [DURATION_SEC]
EXCITATION=free_space DRIVER_VERSION=$(rosversion sr_ronex 2>/dev/null || echo latest) \
  scripts/record_episode.sh 2026_07_02_am ep001 60
```
- Native rates, **no downsampling at record time**. The bag is the source of truth.
- One bag per episode → `data/raw/<session>/<episode>_<ts>.bag` + `.meta.yaml` sidecar.
- Fixed episode length (default 60 s). Keep sessions consistent.

### Excitation regimes to cover (`EXCITATION=`)
- `free_space` — hand moves in air. The bulk of the data.
- `loaded` — fingers press an object / each other (motor works against resistance).
- `manual_perturbation` — operator pushes joints while the controller holds a target.
  This captures the key gap the actuator net must learn: *the motor thinks it's stiff but
  the joint isn't where it commanded.* Include several of these per session.

Record the object / perturbation details in `OPERATOR="..."`.

## 3. Offline processing (re-runnable, never touches the bags)
```bash
python3 preprocess/parse_bag.py data/raw/<session>/*.bag        # QC: rates/drops/jitter
python3 preprocess/align.py      data/raw/<session>/<ep>.bag    # -> data/aligned/...
python3 preprocess/build_dataset.py                            # -> data/dataset/... + manifest
python3 preprocess/normalize.py                                # -> scaler.json (train-only)
python3 qc/report.py     data/aligned/<session>/<ep>.aligned.npz
python3 qc/loader_test.py                                      # contract test
```

## Metadata captured per episode (`meta.yaml`)
date/timestamp, excitation type, control mode (PWM/Torque), warm-up state, driver version,
operator notes, recorded topic list, `ROS_MASTER_URI`.

## Do / Don't
- **Do** re-run pre-flight after any controller restart or e-stop.
- **Do** keep the raw bags; every derived artifact can be rebuilt from them.
- **Don't** collect only a preprocessed format — always keep the bag.
- **Don't** change `config/joints.yaml` ordering without re-running everything downstream.
