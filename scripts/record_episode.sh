#!/usr/bin/env bash
# Record ONE episode to an immutable bag (data/raw/) + sidecar JSON (meta/). The bag is
# the source of truth for signals; the JSON sidecar is the source of truth for what was
# run. Records native rates, no downsampling.
#
# Usage:
#   scripts/record_episode.sh SESSION_ID EPISODE_ID [DURATION_SEC]
# Env overrides:
#   EXCITATION=free_space|loaded|manual_perturbation   (default free_space)
#   CONTROL_MODE=PWM|Torque                            (default PWM)
#   WARMUP=cold|warm                                   (default warm)
#   DRIVER_VERSION=...                                 (default: unknown)
#   OPERATOR="notes..."                                (default empty)
set -euo pipefail

SESSION_ID="${1:?usage: record_episode.sh SESSION_ID EPISODE_ID [DURATION_SEC]}"
EPISODE_ID="${2:?missing EPISODE_ID}"
DURATION="${3:-60}"

EXCITATION="${EXCITATION:-free_space}"
CONTROL_MODE="${CONTROL_MODE:-PWM}"
WARMUP="${WARMUP:-warm}"
DRIVER_VERSION="${DRIVER_VERSION:-unknown}"
OPERATOR="${OPERATOR:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash 2>/dev/null || true
source /ros1/devel/setup.bash 2>/dev/null || true   # touchlab_driver_ros/touchlab_msgs live here

# Tactile: shadow_touchlab_translator publishes the calibrated topic config/topics.yaml
# points "tactile" at (see that file's comment for why calibrated, not raw /rh/tactile).
# It is NOT running by default -- launch it once, idempotently: a multi-episode session
# only pays this cost on the first episode, subsequent ones see it already running.
CALIBRATED_TOPIC="$(python3 - "$REPO" <<'PY'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import config_lib as cl
print(cl.load_topics()["tactile"])
PY
)"
if ! rosnode list 2>/dev/null | grep -q '^/shadow_touchlab_translator$'; then
  echo "Launching shadow_touchlab_translator (tactile calibration node) ..."
  nohup roslaunch touchlab_driver_ros translator.launch \
    > /tmp/shadow_touchlab_translator.log 2>&1 &
  disown
  for _ in $(seq 1 20); do
    rostopic list 2>/dev/null | grep -qF "$CALIBRATED_TOPIC" && break
    sleep 0.5
  done
  if ! rostopic list 2>/dev/null | grep -qF "$CALIBRATED_TOPIC"; then
    echo "ERROR: shadow_touchlab_translator did not start publishing $CALIBRATED_TOPIC" \
         "within 10s -- see /tmp/shadow_touchlab_translator.log" >&2
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$REPO/data/raw/$SESSION_ID"
META_DIR="$REPO/meta/$SESSION_ID"
mkdir -p "$OUTDIR" "$META_DIR"
STEM="${EPISODE_ID}_${TS}"
BAG="$OUTDIR/${STEM}.bag"
META="$META_DIR/${STEM}.json"

# Build the topic list (joint_states + tactile + 13 controller states + context) from config.
mapfile -t TOPICS < <(python3 - "$REPO" <<'PY'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import config_lib as cl
t = cl.load_topics()
topics = [t["joint_states"], t["tactile"]] + list(t["controller_state"]) + list(t.get("context", []))
print("\n".join(topics))
PY
)

echo "Recording ${#TOPICS[@]} topics for ${DURATION}s -> $BAG"
printf '  %s\n' "${TOPICS[@]}"

# --duration stops cleanly; -O sets output; __name avoids node-name clashes.
rosbag record --duration="${DURATION}" -O "$BAG" "${TOPICS[@]}" \
  __name:="record_${SESSION_ID}_${EPISODE_ID}"

# Written via python3 (not a bash heredoc) so operator notes / topic names never need
# manual quoting/escaping to stay valid JSON.
python3 - "$META" "$SESSION_ID" "$EPISODE_ID" "$TS" "$(basename "$BAG")" "$DURATION" \
  "$EXCITATION" "$CONTROL_MODE" "$WARMUP" "$DRIVER_VERSION" "$OPERATOR" \
  "${ROS_MASTER_URI:-}" "${TOPICS[@]}" <<'PY'
import json
import sys

(meta_path, session_id, episode_id, ts, bag, duration, excitation, control_mode,
 warmup, driver_version, operator, ros_master_uri) = sys.argv[1:13]
topics = sys.argv[13:]

meta = {
    "session_id": session_id,
    "episode_id": episode_id,
    "timestamp": ts,
    "bag": bag,
    "duration_sec": int(duration),
    "excitation": excitation,  # free_space | loaded | manual_perturbation
    "control_mode": control_mode,  # PWM | Torque
    "warmup": warmup,  # cold | warm
    "driver_version": driver_version,
    "operator_notes": operator,
    "ros_master_uri": ros_master_uri,
    "recorded_topics": topics,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "wrote $BAG"
echo "wrote $META"
