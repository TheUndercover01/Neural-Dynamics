#!/usr/bin/env bash
# Collect a full raw dataset across excitation regimes via excitation/run_episode.py.
# Only touches the raw-collection stage (bag in data/raw/ + JSON sidecar in meta/) --
# never runs align.py or build_dataset.py itself, so the dataset-building strategy stays
# undecided until later. Velocity-range coverage comes for free from run_episode.py's
# own per-episode random max_delta sampling (excitation/config.py: MAX_DELTA_RANGE) --
# no tuning needed here.
#
# Usage:
#   scripts/collect_dataset.sh SESSION_ID
#   Ctrl-C   (SIGINT)  stops the WHOLE run: kills the in-flight episode, returns the
#            hand home, prints how many episodes were completed, and exits.
#   Ctrl-\   (SIGQUIT) cancels ONLY the in-flight episode: kills it, returns the hand
#            home, then continues on to the next episode/regime in the plan.
#   Both are safe to use mid-episode -- nothing is left dangling either way (see
#   cleanup()/skip_episode() below for why a plain kill of the direct child isn't enough).
#
# Env overrides:
#   REGIMES=free_space free_space_continuous     (default: these 2 only, space-separated --
#                                                  no operator needed, safe to walk away from)
#     Add perturbation / loaded_hold / step_probe explicitly when you're ready for them, e.g.:
#       REGIMES="free_space free_space_continuous perturbation" scripts/collect_dataset.sh SID
#     free_space_continuous has NO steps mixed in -- every one of the 13 joints moves
#     continuously at once, unlike free_space which always carves out a couple of step
#     channels.
#   EPISODES_PER_REGIME=20                       (default: 20, same count for every regime above)
#   DURATION=                                    (default: unset -> each regime's own
#                                                  REGIME_DEFAULT_DURATION_S)
#
# single_joint / range_sweep (add via REGIMES=... same as any other regime) isolate ONE
# actuator per episode -- every other joint held fixed at that episode's start pose --
# and this script auto-cycles which actuator, one per episode (0,1,2,...,12,0,1,...), so
# a run of N episodes gives even per-joint coverage without you passing --joint yourself.
# The cycle position is derived from the same auto-numbered episode index next_index()
# already tracks, so re-running this script for the same SESSION_ID continues the cycle
# rather than restarting at joint 0.
#   single_joint  -- stochastic excitation (ou_walk/multisine/chirp) on the one active
#                    joint; healthy but not guaranteed range coverage per episode.
#   range_sweep   -- deterministic ramp COMMANDING both extremes on the one active
#                    joint every episode. Whether it physically gets there depends on
#                    the other joints' held pose not obstructing it -- confirmed on real
#                    hardware that a jittered neighboring finger can mechanically block
#                    full flexion; act_err correctly records the resulting stall rather
#                    than a false position, so this is informative data, not a failure.
#                    Its duration is auto-computed per joint -- DURATION= is ignored for it.
#
# perturbation/loaded_hold need an operator at the hand (manual object contact /
# perturbation, per COLLECTION_PROTOCOL.md):
#   - a warning is printed ONE EPISODE AHEAD, i.e. during the last unattended episode
#     before a manual regime starts, so you have that episode's whole duration to get
#     in position -- not just an instant before recording starts.
#   - confirmation (tag + Enter) happens ONCE per manual-regime BLOCK, not per episode:
#     the first episode of a run of consecutive perturbation/loaded_hold episodes pauses
#     as before; subsequent episodes of that SAME regime, back to back, run automatically
#     with no further pausing (reusing the same tag). A new pause only happens when the
#     regime changes again (e.g. back to an unattended regime and then into a manual one).
#
# Every episode's full config (regime, families, seeds, max_delta_rad, step_events,
# jitter/overruns, operator_notes, home_pose_actuator, ...) lives in ONE JSON at
# meta/<session>/<episode>.json (preprocess/align.py later merges an "aligned" QC
# section into that same file) -- the episode_id itself also encodes the regime
# (ep_<regime>_NNN), and every per-episode print during collection is prefixed
# "[regime]", so it's always visible on-screen which regime is currently recording.
# Count/group episodes later straight from those files, e.g.:
#   jq -s 'group_by(.regime) | map({regime: .[0].regime, count: length})' \
#     meta/SESSION_ID/*.json
set -euo pipefail

SESSION_ID="${1:?usage: scripts/collect_dataset.sh SESSION_ID}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

read -r -a REGIMES <<< "${REGIMES:-free_space free_space_continuous}"
EPISODES_PER_REGIME="${EPISODES_PER_REGIME:-20}"
DURATION="${DURATION:-}"

declare -A recorded
total=0
n_plan=0
child_pid=""
skip_requested=0
current_episode_id=""

# Kill the in-flight episode's WHOLE process tree. run_episode.py launches
# scripts/record_episode.sh as its own subprocess, which in turn launches `rosbag
# record` as ANOTHER subprocess -- killing just the direct python3 child does NOT
# reach those grandchildren (confirmed: an earlier version of this script left rosbag
# recording, orphaned, for its full --duration after the main script had already
# exited). Each episode is launched via `setsid` below specifically so child_pid IS
# that whole tree's process group id, letting `kill -- -PGID` (negative PID = process
# GROUP) take everything down together: python3, record_episode.sh, and rosbag record
# all at once. Shared by both cleanup() and skip_episode() below.
kill_current_episode() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM -- "-$child_pid" 2>/dev/null
    sleep 1
    kill -KILL -- "-$child_pid" 2>/dev/null || true
  fi
  # The killed episode's own ROS node died with it, so nothing is left commanding the
  # hand -- it's frozen wherever the trajectory was cut off. Return it to the known,
  # safe home pose before doing anything else.
  echo ">>> returning hand to home pose ..."
  python3 excitation/go_home.py 2>&1 || echo "WARNING: failed to return to home pose -- check hand position manually" >&2
}

# Ctrl-C (SIGINT): stop the WHOLE run.
cleanup() {
  echo
  echo ">>> STOPPING (interrupted) -- $total/$n_plan episodes completed before stopping."
  kill_current_episode
  echo -n ">>> final count: "
  for r in "${REGIMES[@]}"; do printf "%s=%d " "$r" "${recorded[$r]:-0}"; done
  echo
  exit 130
}
trap cleanup INT TERM

# Ctrl-\ (SIGQUIT): cancel only the CURRENT episode, then let the main loop continue
# to the next one. Just sets a flag + kills the episode here -- the actual "continue
# to next iteration" happens back in the main loop, after `wait` returns, because a
# `continue` executed from inside a trap does not resume an enclosing shell `for` loop.
skip_episode() {
  echo
  echo ">>> SKIPPING episode ${current_episode_id:-<unknown>} -- the rest of the run will continue."
  skip_requested=1
  kill_current_episode
}
trap skip_episode QUIT

MANUAL_REGIMES=(perturbation loaded_hold)
# Regimes that isolate ONE actuator per episode (require --joint) -- this script
# auto-cycles which actuator across episodes for these, same mechanism for both.
PER_JOINT_REGIMES=(single_joint range_sweep)
is_per_joint() {
  local r="$1" p
  for p in "${PER_JOINT_REGIMES[@]}"; do [[ "$r" == "$p" ]] && return 0; done
  return 1
}
is_manual() {
  local r="$1"
  for m in "${MANUAL_REGIMES[@]}"; do [[ "$r" == "$m" ]] && return 0; done
  return 1
}

echo "=== preflight: scripts/check_stream.py ==="
source /opt/ros/noetic/setup.bash 2>/dev/null || true
python3 scripts/check_stream.py
echo

META_DIR="$REPO/meta/$SESSION_ID"
mkdir -p "$META_DIR"

# actuator_order, for single_joint's auto-cycling below.
mapfile -t ACTUATOR_ORDER < <(python3 - "$REPO" <<'PY'
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import config_lib as cl
print("\n".join(cl.load_joints()["actuator_order"]))
PY
)
N_ACTUATORS=${#ACTUATOR_ORDER[@]}

# Smallest unused 3-digit suffix for ep_<regime>_NNN in this session, so re-running this
# script for the same SESSION_ID appends new episodes instead of colliding with old ones.
next_index() {
  local regime="$1" max=0 base n
  shopt -s nullglob
  for f in "$META_DIR"/ep_${regime}_[0-9][0-9][0-9]_*.json; do
    base="$(basename "$f")"
    n="${base#ep_${regime}_}"; n="${n%%_*}"
    n="$((10#$n))"
    (( n > max )) && max=$n
  done
  shopt -u nullglob
  echo "$max"
}

# --- Build the full plan up front (regime:episode_id per entry) so we can print a
# summary now and peek ONE ENTRY AHEAD during the run to warn before a manual regime. ---
PLAN=()
declare -A regime_count
declare -A PLAN_JOINT   # episode_id -> actuator name, per-joint regimes only (see is_per_joint)
for regime in "${REGIMES[@]}"; do
  regime_count[$regime]=0
  start_idx=$(next_index "$regime")
  for ((i = 1; i <= EPISODES_PER_REGIME; i++)); do
    idx=$((start_idx + i))
    episode_id=$(printf "ep_%s_%03d" "$regime" "$idx")
    PLAN+=("$regime:$episode_id")
    regime_count[$regime]=$((regime_count[$regime] + 1))
    if is_per_joint "$regime"; then
      joint_cycle_idx=$(( (idx - 1) % N_ACTUATORS ))
      PLAN_JOINT["$episode_id"]="${ACTUATOR_ORDER[$joint_cycle_idx]}"
    fi
  done
done
n_plan=${#PLAN[@]}

echo "=== plan: $n_plan episodes ==="
for regime in "${REGIMES[@]}"; do
  tag="unattended"
  is_manual "$regime" && tag="NEEDS OPERATOR (confirm once per block)"
  is_per_joint "$regime" && tag="unattended (cycles all $N_ACTUATORS actuators)"
  printf "  %-22s x%-3d  %s\n" "$regime" "${regime_count[$regime]}" "$tag"
done
echo

prev_regime=""
block_ptype=""
for ((k = 0; k < n_plan; k++)); do
  regime="${PLAN[k]%%:*}"
  episode_id="${PLAN[k]#*:}"

  # One-episode-ahead warning: if the NEXT plan entry is a different, manual regime,
  # flag it now -- while THIS (possibly unattended) episode still has its full
  # duration left to run, giving real lead time to get in position.
  if (( k + 1 < n_plan )); then
    next_regime="${PLAN[k+1]%%:*}"
    if [[ "$next_regime" != "$regime" ]] && is_manual "$next_regime"; then
      echo ">>> HEADS UP: after this episode ($regime), next up is '$next_regime' -- needs an operator at the hand. Get ready during this episode."
    fi
  fi

  ptype=""
  if is_manual "$regime"; then
    if [[ "$regime" != "$prev_regime" ]]; then
      echo ">>> next: $regime block starting at $episode_id -- get in position at the hand."
      read -r -p ">>> perturbation/contact type tag for this whole block (short free text, e.g. finger_push_FFJ3): " block_ptype
      block_ptype="${block_ptype:-unspecified}"
      read -r -p ">>> press Enter to start the $regime block (${regime_count[$regime]} episodes back to back, Ctrl-C to abort the whole run) ..." _
    else
      echo ">>> continuing $regime block: $episode_id (confirmed once at block start, no further pausing)"
    fi
    ptype="$block_ptype"
  fi
  prev_regime="$regime"

  joint_note=""
  dur_args=()
  # DURATION never applies to range_sweep -- its safe duration is computed per-joint
  # from that actuator's own span (excitation/config.py:range_sweep_duration_s()); a
  # global override could be too short to actually reach the extreme, silently
  # truncating the sweep.
  if [[ -n "$DURATION" && "$regime" != "range_sweep" ]]; then
    dur_args=(--duration "$DURATION")
  fi
  if is_per_joint "$regime"; then
    dur_args+=(--joint "${PLAN_JOINT[$episode_id]}")
    joint_note="  joint=${PLAN_JOINT[$episode_id]}"
  fi

  echo "=== [$regime] episode $episode_id ($((total + 1))/$n_plan total)${joint_note} ==="
  echo "    (Ctrl-C = stop the whole run, Ctrl-\\ = skip just this episode)"
  current_episode_id="$episode_id"
  if [[ -n "$ptype" ]]; then
    OPERATOR="$ptype" setsid python3 excitation/run_episode.py --regime "$regime" \
      --session "$SESSION_ID" --episode "$episode_id" "${dur_args[@]}" &
  else
    setsid python3 excitation/run_episode.py --regime "$regime" --session "$SESSION_ID" \
      --episode "$episode_id" "${dur_args[@]}" &
  fi
  child_pid=$!
  # `if wait; then ... else ...` (not a bare `wait`) so a non-zero exit -- whether from
  # skip_episode()'s deliberate kill or a genuine run_episode.py crash -- never trips
  # `set -e` here; the two cases are told apart via skip_requested right below instead,
  # so a real crash still aborts the whole run (unchanged fail-fast behavior) while a
  # deliberate skip does not.
  if wait "$child_pid"; then
    episode_status=0
  else
    episode_status=$?
  fi
  child_pid=""

  if [[ "$skip_requested" == "1" ]]; then
    skip_requested=0
    echo ">>> episode $episode_id skipped -- continuing with the rest of the run."
    echo
    continue
  fi
  if [[ "$episode_status" != "0" ]]; then
    echo "ERROR: episode $episode_id failed (exit $episode_status)" >&2
    exit "$episode_status"
  fi

  meta_file=$(ls -t "$META_DIR/${episode_id}"_*.json 2>/dev/null | head -1)
  if [[ -z "$meta_file" ]]; then
    echo "WARNING: could not find meta/$SESSION_ID JSON for $episode_id" >&2
  fi

  recorded[$regime]=$(( ${recorded[$regime]:-0} + 1 ))
  total=$((total + 1))
  echo -n ">>> progress: "
  for r in "${REGIMES[@]}"; do printf "%s=%d " "$r" "${recorded[$r]:-0}"; done
  echo "(total=$total/$n_plan)  last -> ${meta_file:-<missing>}"
  echo
done

echo "=== done: $total episodes recorded across ${#REGIMES[@]} regimes -> data/raw/$SESSION_ID + meta/$SESSION_ID ==="
