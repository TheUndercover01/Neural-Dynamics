#!/usr/bin/env python3
"""CLI: compose an excitation episode for one regime, record it, and stamp meta.yaml.

    python3 excitation/run_episode.py --regime step_probe --dry-run
    python3 excitation/run_episode.py --regime free_space --session 2026_07_09_pm --episode ep001

--dry-run composes the trajectory and prints its schedule/step_events WITHOUT touching
ROS or rosbag — runnable on a box with no ROS install, for CI/offline sanity-checking.

The real run reuses scripts/record_episode.sh (rosbag record on the native-rate topic
list, writes the meta.yaml base) rather than reimplementing rosbag record here. That
script's `rosbag record --duration=...` call is BLOCKING and writes meta.yaml only
AFTER it exits, so this script launches it as a background subprocess, runs the publish
loop concurrently while it's recording, waits for it to exit, then APPENDS excitation-
specific keys to the meta.yaml it already wrote (never clobbering the base).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402
import compose  # noqa: E402

REGIMES = list(ecfg.REGIME_FAMILIES.keys())


def _print_schedule(regime: str, traj: np.ndarray, step_events: list[dict], info: dict,
                     rate: float) -> None:
    joints = cl.load_joints()
    acts = joints["actuator_order"]
    T = traj.shape[0]
    print(f"regime={regime}  T={T} frames  duration={T / rate:.2f}s  rate={rate:g}Hz")
    print(f"families={info['families']}  family_seeds={info['family_seeds']}")
    print(f"steps_channels={[acts[i] for i in info['steps_channels']]}")
    print(f"continuous_channels={[acts[i] for i in info['continuous_channels']]}")
    print(f"max_delta_rad={info['max_delta_rad']:.4f}")
    print(f"n_step_events={len(step_events)}")
    for e in step_events[:10]:
        print(f"  frame={e['frame']:5d} actuator={acts[e['actuator_idx']]:10s} "
              f"pre={e['pre']:+.4f} post={e['post']:+.4f} mag={e['magnitude']:+.4f}")
    if len(step_events) > 10:
        print(f"  ... and {len(step_events) - 10} more")


def _build_episode(args) -> tuple[np.ndarray, list[dict], dict]:
    families_weights = ecfg.REGIME_FAMILIES[args.regime]
    duration_s = args.duration or ecfg.REGIME_DEFAULT_DURATION_S[args.regime]
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31 - 1))
    traj, step_events, info = compose.compose_episode(
        args.regime, list(families_weights.keys()), families_weights,
        rng_seed=seed, duration_s=duration_s, rate=ecfg.PUBLISH_RATE_HZ)
    info["rng_seed"] = seed
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
            elif p.suffixes[-2:] == [".meta", ".yaml"]:
                paths["meta"] = p
    return paths


def _append_meta(meta_path: pathlib.Path, traj: np.ndarray, step_events: list[dict],
                  info: dict, play_result: dict, rate: float) -> None:
    joints = cl.load_joints()
    acts = joints["actuator_order"]
    meta = yaml.safe_load(meta_path.read_text()) or {}
    meta.update({
        "regime": info["regime"],
        "families": info["families"],
        "rng_seed": info["rng_seed"],
        "family_seeds": info["family_seeds"],
        "command_space": "actuator_order",  # radians, raw to /command, no 2x multiplier
        "publish_rate_hz": rate,
        "max_delta_rad": info["max_delta_rad"],
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
    })
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--session", default=None, help="required unless --dry-run")
    ap.add_argument("--episode", default=None, help="required unless --dry-run")
    ap.add_argument("--seed", type=int, default=None, help="default: random")
    ap.add_argument("--duration", type=float, default=None,
                     help=f"default: excitation.config.REGIME_DEFAULT_DURATION_S[regime]")
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
    record_dur = 2 * ecfg.LEAD_IN_S + duration_s + ecfg.LEAD_OUT_S + ecfg.RECORD_MARGIN_S
    excitation_tag = ecfg.REGIME_TO_EXCITATION[args.regime]

    print(f"\nlaunching scripts/record_episode.sh {args.session} {args.episode} "
          f"{record_dur:.0f}  (EXCITATION={excitation_tag})")
    proc = _run_record_episode_sh(args.session, args.episode, record_dur, excitation_tag)

    import rospy
    rospy.init_node("run_episode", anonymous=True, disable_signals=True)
    pubs = publisher.create_publishers()
    rospy.sleep(0.5)  # let publishers connect
    time.sleep(ecfg.LEAD_IN_S)  # give rosbag time to advertise/subscribe before we publish

    print("reading current actuator pose ...")
    current = publisher.read_current_actuator_pose()
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
        print("ERROR: could not find meta.yaml path in record_episode.sh output; "
              "excitation metadata NOT appended.", file=sys.stderr)
        return 1

    _append_meta(paths["meta"], traj, step_events, info, play_result, rate)
    print(f"appended excitation metadata to {paths['meta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
