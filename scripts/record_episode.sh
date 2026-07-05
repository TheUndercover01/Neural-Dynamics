#!/usr/bin/env bash
# Record ONE episode to an immutable bag + sidecar meta.yaml.
# Records native rates, no downsampling. The bag is the source of truth.
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

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$REPO/data/raw/$SESSION_ID"
mkdir -p "$OUTDIR"
PREFIX="$OUTDIR/${EPISODE_ID}_${TS}"
BAG="${PREFIX}.bag"
META="${PREFIX}.meta.yaml"

# Build the topic list (joint_states + 13 controller states + context) from config.
mapfile -t TOPICS < <(python3 - "$REPO" <<'PY'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import config_lib as cl
t = cl.load_topics()
topics = [t["joint_states"]] + list(t["controller_state"]) + list(t.get("context", []))
print("\n".join(topics))
PY
)

echo "Recording ${#TOPICS[@]} topics for ${DURATION}s -> $BAG"
printf '  %s\n' "${TOPICS[@]}"

# --duration stops cleanly; -O sets output; __name avoids node-name clashes.
rosbag record --duration="${DURATION}" -O "$BAG" "${TOPICS[@]}" \
  __name:="record_${SESSION_ID}_${EPISODE_ID}"

cat > "$META" <<EOF
session_id: $SESSION_ID
episode_id: $EPISODE_ID
timestamp: $TS
bag: $(basename "$BAG")
duration_sec: $DURATION
excitation: $EXCITATION            # free_space | loaded | manual_perturbation
control_mode: $CONTROL_MODE        # PWM | Torque
warmup: $WARMUP                    # cold | warm
driver_version: "$DRIVER_VERSION"
operator_notes: "$OPERATOR"
ros_master_uri: "${ROS_MASTER_URI:-}"
recorded_topics:
$(printf '  - %s\n' "${TOPICS[@]}")
EOF

echo "wrote $BAG"
echo "wrote $META"
