#!/usr/bin/env bash
# Capture live topology + rates from the hand into DISCOVERY.generated.md.
# Run this ON A NODE ATTACHED TO THE HAND'S NETWORK (not a container that can't route to
# the controller publishers). It never touches the curated DISCOVERY.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/DISCOVERY.generated.md"
HZ_SECS="${HZ_SECS:-8}"   # seconds to sample each rate

source /opt/ros/noetic/setup.bash 2>/dev/null || true

if ! rostopic list >/dev/null 2>&1; then
  echo "ERROR: no ROS master reachable (ROS_MASTER_URI=$ROS_MASTER_URI)" >&2
  exit 1
fi

# Pull the topic list the pipeline cares about from topics.yaml via config_lib.
mapfile -t STATE_TOPICS < <(python3 - "$REPO" <<'PY'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import config_lib as cl
for t in cl.actuator_state_topics():
    print(t)
PY
)

{
  echo "# DISCOVERY (generated)"
  echo
  echo "_Captured $(date -Iseconds) — ROS_MASTER_URI=$ROS_MASTER_URI_"
  echo

  echo "## Controller state topics present"
  echo '```'
  rostopic list | grep -E 'position_controller/state$' | sort
  echo '```'
  n=$(rostopic list | grep -cE 'position_controller/state$' || true)
  echo "count = $n (expected 13)"
  echo

  echo "## Message types"
  echo '```'
  echo "/joint_states -> $(rostopic type /joint_states)"
  echo "${STATE_TOPICS[0]} -> $(rostopic type "${STATE_TOPICS[0]}")"
  echo '```'
  echo

  echo "## Measured rates (${HZ_SECS}s window)"
  echo '```'
  for t in /joint_states "${STATE_TOPICS[0]}" /diagnostics; do
    rate=$(timeout "$HZ_SECS" rostopic hz "$t" 2>/dev/null | grep -m1 'average rate' || echo "average rate: (no data)")
    printf '%-48s %s\n' "$t" "$rate"
  done
  echo '```'
  echo

  echo "## Control mode (PID params for first actuator)"
  echo '```'
  rosparam list 2>/dev/null | grep -iE '/rh/ffj0/pid/' || echo "(no pid params found)"
  echo '```'
  echo "Presence of max_pwm + sg_* => PWM position control."
  echo

  echo "## One /joint_states sample (verify 16-joint order)"
  echo '```'
  timeout 5 rostopic echo -n1 /joint_states 2>/dev/null | sed -n '1,60p'
  echo '```'
} > "$OUT"

echo "wrote $OUT"
