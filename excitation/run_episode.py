#!/usr/bin/env python3
"""CLI: compose an excitation episode for one regime, record it, and stamp its meta/ JSON.

    python3 excitation/run_episode.py --regime step_probe --dry-run
    python3 excitation/run_episode.py --regime free_space --session 2026_07_09_pm --episode ep001

--dry-run composes the trajectory and prints its schedule/step_events WITHOUT touching
ROS or rosbag — runnable on a box with no ROS install, for CI/offline sanity-checking.

The real run reuses scripts/record_episode.sh (rosbag record on the native-rate topic
list, writes the meta/<session>/<episode>.json base) rather than reimplementing rosbag
record here. That script's `rosbag record --duration=...` call is BLOCKING and writes
the JSON sidecar only AFTER it exits, so this script launches it as a background
subprocess, runs the publish loop concurrently while it's recording, waits for it to
exit, then APPENDS excitation-specific keys to the JSON it already wrote (never
clobbering the base). preprocess/align.py later merges an "aligned" section into this
same file -- one JSON per episode ends up describing everything about it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402
import compose  # noqa: E402

REGIMES = list(ecfg.REGIME_FAMILIES.keys())


def _resolve_joint(joint_arg: str, acts: list[str]) -> int:
    """--joint accepts an exact actuator name (e.g. rh_FFJ3) or a numeric index into
    actuator_order (0-12)."""
    if joint_arg in acts:
        return acts.index(joint_arg)
    try:
        idx = int(joint_arg)
    except ValueError:
        raise SystemExit(f"--joint {joint_arg!r} not recognized. Use an exact actuator "
                          f"name or an index 0-{len(acts) - 1}. actuator_order: {acts}")
    if not (0 <= idx < len(acts)):
        raise SystemExit(f"--joint index {idx} out of range 0-{len(acts) - 1}. "
                          f"actuator_order: {acts}")
    return idx


def _print_schedule(regime: str, traj: np.ndarray, step_events: list[dict], info: dict,
                     rate: float) -> None:
    joints = cl.load_joints()
    acts = joints["actuator_order"]
    T = traj.shape[0]
    print(f"regime={regime}  T={T} frames  duration={T / rate:.2f}s  rate={rate:g}Hz")
    print(f"families={info['families']}  family_seeds={info['family_seeds']}")
    print(f"steps_channels={[acts[i] for i in info['steps_channels']]}")
    print(f"continuous_channels={[acts[i] for i in info['continuous_channels']]}")
    for f, idxs in info.get("continuous_family_channels", {}).items():
        print(f"  {f}: {[acts[i] for i in idxs]}")
    print(f"max_delta_rad={info['max_delta_rad']:.4f}")
    print(f"n_step_events={len(step_events)}")
    for e in step_events[:10]:
        print(f"  frame={e['frame']:5d} actuator={acts[e['actuator_idx']]:10s} "
              f"pre={e['pre']:+.4f} post={e['post']:+.4f} mag={e['magnitude']:+.4f}")
    if len(step_events) > 10:
        print(f"  ... and {len(step_events) - 10} more")


def _build_episode(args) -> tuple[np.ndarray, list[dict], dict]:
    families_weights = ecfg.REGIME_FAMILIES[args.regime]
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31 - 1))

    # This episode's actual pre-trajectory reset/settle pose -- home_pose_actuator()
    # jittered per joint (excitation.config.START_POSE_JITTER_FRAC), NOT the same fixed
    # pose every episode. Deterministic from `seed` via its own Generator instance, so it
    # doesn't consume/perturb compose_episode()'s own internal rng_seed-derived draws.
    start_pose = ecfg.randomized_start_pose(seed)

    active_channels = None
    inactive_value = None
    if args.joint is not None:
        acts = cl.load_joints()["actuator_order"]
        active_channels = [_resolve_joint(args.joint, acts)]
        # Held channels stay at THIS episode's start_pose, not the generic joint midpoint
        # -- every episode already resets+settles there before this trajectory starts
        # (see main()'s HOME_SETTLE_S step), so this means those joints genuinely don't
        # move at all for the whole episode, not just "hold some arbitrary point".
        inactive_value = start_pose
    elif args.regime == "range_sweep":
        raise SystemExit("--regime range_sweep requires --joint -- it's a per-actuator "
                          "full-range sweep; use scripts/collect_dataset.sh's auto-cycling "
                          "to cover all 13 actuators across episodes.")

    if args.regime == "range_sweep":
        # NOT the usual random DURATION_RANGE_S draw -- a random duration could be too
        # short to safely reach the true extreme, silently truncating the sweep. See
        # excitation/config.py:range_sweep_duration_s().
        duration_s = args.duration or ecfg.range_sweep_duration_s(active_channels[0])
    else:
        # Drawn from a Generator seeded with `seed` (not args.duration's own separate RNG
        # state) so the same --seed reproduces the same duration_s too -- see
        # excitation/config.py:DURATION_RANGE_S. A fresh Generator here doesn't consume any
        # of compose_episode()'s own internal rng_seed-derived draws (it creates its own),
        # so this doesn't perturb channel allocation / family_seeds / max_delta reproducibility.
        duration_s = args.duration or float(np.random.default_rng(seed).uniform(*ecfg.DURATION_RANGE_S))

    traj, step_events, info = compose.compose_episode(
        args.regime, list(families_weights.keys()), families_weights,
        rng_seed=seed, duration_s=duration_s, rate=ecfg.PUBLISH_RATE_HZ,
        active_channels=active_channels, inactive_value=inactive_value)
    info["rng_seed"] = seed
    info["duration_s"] = duration_s
    info["start_pose"] = start_pose
    return traj, step_events, info


def _run_record_episode_sh(session: str, episode: str, record_dur: float, excitation_tag: str):
    """Launch scripts/record_episode.sh as a background subprocess. Returns the Popen
    handle; caller must wait() on it and then parse its captured stdout for the
    'wrote <path>' lines it prints (bag path, then meta path)."""
    import os
    env = os.environ.copy()
    env["EXCITATION"] = excitation_tag
    script = cl.REPO_ROOT / "scripts" / "record_episode.sh"
    proc = subprocess.Popen(
        [str(script), session, episode, str(int(round(record_dur)))],
        cwd=str(cl.REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc


def _parse_wrote_paths(stdout_text: str) -> dict:
    paths = {}
    for line in stdout_text.splitlines():
        if line.startswith("wrote "):
            p = pathlib.Path(line[len("wrote "):].strip())
            if p.suffix == ".bag":
                paths["bag"] = p
            elif p.suffix == ".json":
                paths["meta"] = p
    return paths


def _append_meta(meta_path: pathlib.Path, traj: np.ndarray, step_events: list[dict],
                  info: dict, play_result: dict, rate: float, setpoint_gap: dict) -> None:
    joints = cl.load_joints()
    acts = joints["actuator_order"]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "regime": info["regime"],
        "families": info["families"],
        "rng_seed": info["rng_seed"],
        # Pure trajectory duration (compose_episode()'s duration_s arg) -- NOT the same
        # as duration_sec below, which is the full recorded bag length (home-settle +
        # lead-in ramp + this + lead-out + margin). Needed to regenerate the exact same
        # trajectory from rng_seed alone, since duration_s is now itself randomly drawn
        # per episode (excitation.config.DURATION_RANGE_S) rather than a fixed per-regime
        # constant.
        "duration_s": info["duration_s"],
        "family_seeds": info["family_seeds"],
        "command_space": "actuator_order",  # radians, raw to /command, no 2x multiplier
        "publish_rate_hz": rate,
        "max_delta_rad": info["max_delta_rad"],
        # Which continuous family (ou_walk/multisine/chirp) exclusively owns each
        # channel this episode, and the per-channel anchor point they oscillate
        # around -- both randomized per episode, see excitation/compose.py.
        "continuous_family_channels": {
            f: [acts[i] for i in idxs]
            for f, idxs in info.get("continuous_family_channels", {}).items()
        },
        "center": dict(zip(acts, info.get("center", []))),
        # Set when --joint restricted this episode to one actuator (e.g. single_joint
        # regime); None for a normal multi-joint episode.
        "target_joint": (acts[info["active_channels"][0]]
                          if info.get("active_channels") else None),
        "loop_jitter_ms": play_result["jitter_ms"],
        "loop_overruns": play_result["overruns"],
        "per_channel_delta_max": dict(zip(acts, info["per_channel_delta_max"])),
        "per_channel_delta_p95": dict(zip(acts, info["per_channel_delta_p95"])),
        "step_events": [
            {"frame": e["frame"], "actuator": acts[e["actuator_idx"]],
             "pre": e["pre"], "post": e["post"], "magnitude": e["magnitude"]}
            for e in step_events
        ],
        "native_rates_hz": ecfg.NATIVE_RATES_HZ,
        # The canonical baoding-ball reference pose (policy->actuator conversion, coupled
        # joints x2, confirmed convention; clipped to effective_command_limits()) -- NOT
        # what this episode actually reset to, see episode_start_pose below.
        "home_pose_actuator": dict(zip(acts, ecfg.home_pose_actuator().tolist())),
        # What this episode ACTUALLY reset to and settled at for home_settle_s before
        # playing: home_pose_actuator() jittered per joint by up to
        # +-start_pose_jitter_frac of that joint's span (excitation/config.py:
        # randomized_start_pose()) -- different every episode by design.
        "episode_start_pose": dict(zip(acts, info["start_pose"].tolist())),
        "start_pose_jitter_frac": ecfg.START_POSE_JITTER_FRAC,
        "home_settle_s": ecfg.HOME_SETTLE_S,
        # Snapshot taken right after the home-settle hold, before play() starts (see
        # publisher.read_current_setpoint_gap): a large max_gap_rad here means the
        # controller DIDN'T settle to home cleanly (worth investigating), or -- for a
        # gap measured before this reset existed -- predicts a one-time "catch-up" jump
        # in the recorded action trace. Real, physically-grounded, not a resampling
        # artifact -- see COLLECTION_PROTOCOL.md.
        "pre_episode_setpoint_gap": setpoint_gap,
    })
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--session", default=None, help="required unless --dry-run")
    ap.add_argument("--episode", default=None, help="required unless --dry-run")
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--duration", type=float, default=None,
                     help="default: random, uniform in excitation.config.DURATION_RANGE_S "
                          "(seeded off --seed, so the draw is reproducible)")
    ap.add_argument("--joint", default=None,
                     help="restrict this episode to ONE actuator (exact name e.g. "
                          "rh_FFJ3, or an index 0-12 into actuator_order) -- every "
                          "other joint stays fixed at this episode's start pose for the "
                          "whole episode. Works with any regime, not just single_joint / "
                          "range_sweep (which are the regimes designed around it, see "
                          "config.py) -- range_sweep REQUIRES it.")
    ap.add_argument("--dry-run", action="store_true",
                     help="compose + print the schedule only, no ROS/rosbag")
    args = ap.parse_args()

    traj, step_events, info = _build_episode(args)
    rate = ecfg.PUBLISH_RATE_HZ
    _print_schedule(args.regime, traj, step_events, info, rate)

    if args.dry_run:
        print("\n--dry-run: no ROS/rosbag touched. jitter: N/A (publish loop did not run).")
        return 0

    if not args.session or not args.episode:
        raise SystemExit("--session and --episode are required unless --dry-run")

    import publisher  # local import: only needed on the real (ROS) path

    duration_s = traj.shape[0] / rate
    # HOME_SETTLE_S (drive to + settle at home_pose_actuator()) replaces the old silent
    # "give rosbag time to advertise/subscribe" wait -- it's a superset (also publishes,
    # so rosbag's subscription completes during it same as before) plus it's where the
    # pre-episode reset actually happens. LEAD_IN_S below is play()'s OWN internal ramp
    # from home to traj[0].
    record_dur = ecfg.HOME_SETTLE_S + ecfg.LEAD_IN_S + duration_s + ecfg.LEAD_OUT_S + ecfg.RECORD_MARGIN_S
    excitation_tag = ecfg.REGIME_TO_EXCITATION[args.regime]

    print(f"\nlaunching scripts/record_episode.sh {args.session} {args.episode} "
          f"{record_dur:.0f}  (EXCITATION={excitation_tag})")
    proc = _run_record_episode_sh(args.session, args.episode, record_dur, excitation_tag)

    import rospy
    rospy.init_node("run_episode", anonymous=True, disable_signals=True)
    pubs = publisher.create_publishers()
    rospy.sleep(0.5)  # let publishers connect

    home_pose = info["start_pose"]
    print(f"resetting to this episode's start pose (baoding-ball jittered "
          f"+-{ecfg.START_POSE_JITTER_FRAC:.0%}), settling {ecfg.HOME_SETTLE_S:.1f}s ...")
    publisher.hold(pubs, home_pose, ecfg.HOME_SETTLE_S, rate)

    print("reading current actuator pose ...")
    setpoint_gap = publisher.read_current_setpoint_gap()
    acts = cl.load_joints()["actuator_order"]
    current = np.array([setpoint_gap["process_value"][a] for a in acts], dtype=np.float64)
    if setpoint_gap["max_gap_rad"] > 0.1:
        worst = max(setpoint_gap["gap"], key=setpoint_gap["gap"].get)
        print(f"NOTE: set_point/process_value gap up to {setpoint_gap['max_gap_rad']:.4f} rad "
              f"(worst: {worst}) even AFTER settling at home -- controller may be tracking "
              f"poorly; expect a one-time 'catch-up' jump at the start of the recorded "
              f"action trace.")
    print("playing trajectory ...")
    play_result = publisher.play(pubs, traj, rate, lead_in_ramp_from=current)
    print(f"play() done: jitter_ms={play_result['jitter_ms']} overruns={play_result['overruns']}")
    publisher.hold(pubs, traj[-1], ecfg.LEAD_OUT_S, rate)

    print("waiting for record_episode.sh (rosbag) to finish ...")
    stdout_text, _ = proc.communicate()
    print(stdout_text)
    if proc.returncode != 0:
        print(f"WARNING: record_episode.sh exited with code {proc.returncode}", file=sys.stderr)

    paths = _parse_wrote_paths(stdout_text)
    if "meta" not in paths:
        print("ERROR: could not find the JSON sidecar path in record_episode.sh output; "
              "excitation metadata NOT appended.", file=sys.stderr)
        return 1

    _append_meta(paths["meta"], traj, step_events, info, play_result, rate, setpoint_gap)
    print(f"appended excitation metadata to {paths['meta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
