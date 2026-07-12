#!/usr/bin/env python3
"""Command the hand to the baoding-ball home pose and hold it there. Standalone entry
point, separate from run_episode.py, specifically so it can be run AFTER an interrupt
has already killed an in-flight episode's own ROS node/process (see
scripts/collect_dataset.sh's cleanup() trap) -- without this, an interrupted episode
leaves the hand frozen wherever the trajectory was cut off, rather than in the known,
safe home position every episode normally starts from.

Ramps smoothly from wherever the hand currently is (read live, not assumed) to
home_pose_actuator(), same lead-in-ramp safety used at the start of every episode, then
holds at home for --hold-s.

    python3 excitation/go_home.py [--hold-s 4.0]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hold-s", type=float, default=ecfg.HOME_SETTLE_S,
                     help=f"seconds to hold at home after arriving (default {ecfg.HOME_SETTLE_S})")
    args = ap.parse_args()

    import publisher
    import rospy

    rospy.init_node("go_home", anonymous=True, disable_signals=True)
    pubs = publisher.create_publishers()
    rospy.sleep(0.5)  # let publishers connect

    rate = ecfg.PUBLISH_RATE_HZ
    home = ecfg.home_pose_actuator()

    print("reading current actuator pose ...")
    current = publisher.read_current_actuator_pose()
    print(f"ramping to home pose over {ecfg.LEAD_IN_S:.1f}s, then holding {args.hold_s:.1f}s ...")

    n_hold = max(1, int(round(args.hold_s * rate)))
    hold_traj = np.tile(home, (n_hold, 1))
    play_result = publisher.play(pubs, hold_traj, rate, lead_in_ramp_from=current)
    print(f"done. jitter_ms={play_result['jitter_ms']} overruns={play_result['overruns']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
