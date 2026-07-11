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
#
# Env overrides:
#   REGIMES=free_space free_space_continuous     (default: these 2 only, space-separated --
#                                                  no operator needed, safe to walk away from)
#     Add perturbation / loaded_hold / step_probe explicitly when you're ready for them, e.g.:
#       REGIMES="free_space free_space_continuous perturbation" scripts/collect_dataset.sh SID
#     free_space_continuous has NO steps mixed in -- every one of the 13 joints moves
#     continuously at once, unlike free_space which always carves out a couple of step
#     channels.
#   EPISODES_PER_REGIME=5                        (default: 5, same count for every regime above)
#   DURATION=                                    (default: unset -> each regime's own
#                                                  REGIME_DEFAULT_DURATION_S)
#
# perturbation/loaded_hold need an operator at the hand (manual object contact /
# perturbation, per COLLECTION_PROTOCOL.md):
#   - a warning is printed ONE EPISODE AHEAD, i.e. during the last unattended episode
#     before a manual regime starts, so you have that episode's whole duration to get
#     in position -- not just an instant before recording starts.
#   - the script then pauses for Enter before each manual episode, and prompts for a
#     short free-text type/contact tag (e.g. finger_push_FFJ3, object_squeeze),
#     forwarded as OPERATOR to run_episode.py -- ends up in that episode's
#     meta/<session>/<episode>.json under operator_notes, no fixed taxonomy needed.
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

MANUAL_REGIMES=(perturbation loaded_hold)
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
for regime in "${REGIMES[@]}"; do
  regime_count[$regime]=0
  start_idx=$(next_index "$regime")
  for ((i = 1; i <= EPISODES_PER_REGIME; i++)); do
    idx=$((start_idx + i))
    PLAN+=("$regime:$(printf "ep_%s_%03d" "$regime" "$idx")")
    regime_count[$regime]=$((regime_count[$regime] + 1))
  done
done

echo "=== plan: ${#PLAN[@]} episodes ==="
for regime in "${REGIMES[@]}"; do
  tag="unattended"
  is_manual "$regime" && tag="NEEDS OPERATOR"
  printf "  %-22s x%-3d  %s\n" "$regime" "${regime_count[$regime]}" "$tag"
done
echo

declare -A recorded
total=0
n_plan=${#PLAN[@]}
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
    echo ">>> next: $regime episode $episode_id -- get in position at the hand."
    read -r -p ">>> perturbation/contact type tag (short free text, e.g. finger_push_FFJ3): " ptype
    ptype="${ptype:-unspecified}"
    read -r -p ">>> press Enter to start recording (Ctrl-C to abort the whole run) ..." _
  fi

  echo "=== [$regime] episode $episode_id ($((total + 1))/$n_plan total) ==="
  dur_args=()
  [[ -n "$DURATION" ]] && dur_args=(--duration "$DURATION")
  if [[ -n "$ptype" ]]; then
    OPERATOR="$ptype" python3 excitation/run_episode.py --regime "$regime" \
      --session "$SESSION_ID" --episode "$episode_id" "${dur_args[@]}"
  else
    python3 excitation/run_episode.py --regime "$regime" --session "$SESSION_ID" \
      --episode "$episode_id" "${dur_args[@]}"
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
