#!/usr/bin/env python3
"""Command ONE joint a clean step (hold -> jump -> hold) at a fixed rate, for
step-response data collection (see actuator_data/COLLECTION_PROTOCOL.md and
actuator_data/diagnostics/latency_timeconstant.py).

Reuses run_simgcan u ap_hardware.py's publishers/scaling directly (same directory import) so
the command path — topic names, the coupled-J0 2x multiplier, limits — is identical to
what run_simgap already uses on this hand.

IMPORTANT: start `rosbag record` (e.g. actuator_data/scripts/record_episode.sh) BEFORE
running this script, with a couple seconds of margin before/after. This script only
publishes commands — it does not record anything itself.

Usage:
    python3 send_step.py --joint rh_FFJ3 --target 0.9
    python3 send_step.py --joint rh_FFJ3 --target-frac 0.6 --settle_s 2 --hold_s 1.5
    python3 send_step.py --joint 4 --target 0.9          # policy index instead of name
    python3 send_step.py --joint rh_FFJ2 --target-frac 0.8   # coupled J0 path (rh_FFJ0)

Prints the exact step frame/time so it lines up with what parse_bag.py/align.py will
later report for the same event.
"""
from __future__ import annotations

import argparse
import sys
from threading import Lock

import numpy as np
import rospy
from control_msgs.msg import JointControllerState

from run_simgap_hardware import (
    DEFAULT_JOINT_POS,
    LOWER_LIMITS,
    POLICY_JOINT_ORDER,
    UPPER_LIMITS,
    create_hand_publishers,
    publish_to_hand,
    scale,
)

# actuator (as in create_hand_publishers' controller_names) -> policy index, and whether
# it needs the /2.0 coupling-multiplier reversal (see publish_to_hand's ffj0/mfj0/rfj0 *2.0).
_ACTUATOR_TO_POLICY = {
    "ffj4": (0, False), "mfj4": (1, False), "rfj4": (2, False), "thj5": (3, False),
    "ffj3": (4, False), "mfj3": (5, False), "rfj3": (6, False), "thj4": (7, False),
    "ffj0": (8, True), "mfj0": (9, True), "rfj0": (10, True),
    "thj2": (11, False), "thj1": (12, False),
}


def read_current_pose(timeout_s: float = 3.0) -> np.ndarray:
    """Reads the hand's CURRENT live process_value into a 13-vector, policy order.

    Coupled actuators (ffj0/mfj0/rfj0) are divided by 2.0 to invert publish_to_hand's
    2x multiplier, so this is directly usable as a hold pose passed back into
    publish_to_hand() without moving those joints.
    """
    lock = Lock()
    latest: dict[str, float] = {}

    def _mk_cb(name):
        def _cb(msg):
            with lock:
                latest[name] = msg.process_value
        return _cb

    subs = []
    for name in _ACTUATOR_TO_POLICY:
        topic = f"/sh_rh_{name}_position_controller/state"
        subs.append(rospy.Subscriber(topic, JointControllerState, _mk_cb(name), queue_size=5))

    deadline = rospy.Time.now().to_sec() + timeout_s
    rate = rospy.Rate(20)
    while rospy.Time.now().to_sec() < deadline and not rospy.is_shutdown():
        with lock:
            if len(latest) == len(_ACTUATOR_TO_POLICY):
                break
        rate.sleep()
    for s in subs:
        s.unregister()

    with lock:
        missing = [n for n in _ACTUATOR_TO_POLICY if n not in latest]
        if missing:
            raise RuntimeError(f"timed out waiting for current pose from: {missing} "
                              f"(controller /state topics not publishing?)")
        pose = np.zeros(13, dtype=np.float32)
        for name, (pol_idx, coupled) in _ACTUATOR_TO_POLICY.items():
            pose[pol_idx] = latest[name] / 2.0 if coupled else latest[name]
    return pose

# Mirrors actuator_data/config/latency.yaml's step_thresh (0.1 rad) — a step smaller
# than this won't register as a "step" to the latency diagnostic's detector.
MIN_STEP_RAD = 0.1


def resolve_joint(joint_arg: str) -> int:
    """Accepts either a policy index ('4') or an exact policy-order joint name."""
    try:
        idx = int(joint_arg)
        if not (0 <= idx < len(POLICY_JOINT_ORDER)):
            raise ValueError
        return idx
    except ValueError:
        pass
    if joint_arg in POLICY_JOINT_ORDER:
        return POLICY_JOINT_ORDER.index(joint_arg)
    raise SystemExit(
        f"--joint {joint_arg!r} not recognized. Use a policy index 0-12 or one of: "
        f"{POLICY_JOINT_ORDER}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", required=True,
                    help="policy index (0-12) or exact name, e.g. rh_FFJ3 / rh_FFJ2 (coupled)")
    ap.add_argument("--start", type=float, default=None,
                    help="radians to hold before the step (default: the hold pose's own "
                         "value for this joint)")
    tgt = ap.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target", type=float, default=None, help="radians to jump to")
    tgt.add_argument("--target-frac", type=float, default=None,
                     help="jump target as a fraction in [-1,1] of this joint's [lower,upper] "
                          "range (same convention as the policy's own action space)")
    ap.add_argument("--settle_s", type=float, default=1.5,
                    help="seconds to hold --start before the jump (default 1.5)")
    ap.add_argument("--hold_s", type=float, default=1.0,
                    help="seconds to hold --target after the jump (default 1.0 = 60 "
                         "frames at 60Hz, well over the pipeline's hold_frames=20)")
    ap.add_argument("--pose", choices=["current", "zero", "default"], default="current",
                    help="baseline pose for the other 12 joints while this one steps. "
                         "current (default, SAFE) = read the hand's live position first and "
                         "hold everything else exactly there; zero = all-zero pose "
                         "(WILL MOVE every other joint if the hand isn't already at zero); "
                         "default = the Baoding-ball pose (WILL MOVE every other joint "
                         "unless the hand is already holding it)")
    ap.add_argument("--rate", type=float, default=60.0,
                    help="publish rate Hz (default 60, matches actuator_data's dataset_rate)")
    args = ap.parse_args()

    idx = resolve_joint(args.joint)
    name = POLICY_JOINT_ORDER[idx]
    lower, upper = float(LOWER_LIMITS[idx]), float(UPPER_LIMITS[idx])

    rospy.init_node("send_step", anonymous=True, disable_signals=True)

    if args.pose == "current":
        print("Reading current hand pose (so the other 12 joints don't move)...")
        pose = read_current_pose()
        print("  current pose (policy order):", np.array2string(pose, precision=4))
    else:
        pose = (DEFAULT_JOINT_POS if args.pose == "default" else np.zeros(13, dtype=np.float32)).copy()
        print(f"WARNING: --pose {args.pose} does NOT read the hand's live position — "
              f"every joint other than {POLICY_JOINT_ORDER[idx]} will jump to this fixed "
              f"pose the instant this script starts publishing, regardless of where the "
              f"hand currently is.", file=sys.stderr)

    start = args.start if args.start is not None else float(pose[idx])
    if args.target_frac is not None:
        if not -1.0 <= args.target_frac <= 1.0:
            raise SystemExit(f"--target-frac must be in [-1,1], got {args.target_frac}")
        target = float(scale(np.array([args.target_frac]), np.array([lower]), np.array([upper]))[0])
    else:
        target = args.target

    start_c, target_c = np.clip([start, target], lower, upper)
    if start_c != start or target_c != target:
        print(f"WARNING: clipped to [{lower:.4f}, {upper:.4f}] rad: "
              f"start {start:.4f}->{start_c:.4f}, target {target:.4f}->{target_c:.4f}",
              file=sys.stderr)
    start, target = float(start_c), float(target_c)

    if abs(target - start) <= MIN_STEP_RAD:
        raise SystemExit(f"|target-start| = {abs(target-start):.4f} rad <= MIN_STEP_RAD="
                         f"{MIN_STEP_RAD} — the latency diagnostic's step detector "
                         f"(step_thresh in config/latency.yaml) would not see this as a step. "
                         f"Pick a bigger --target/--target-frac.")

    n_settle = int(round(args.settle_s * args.rate))
    n_hold = int(round(args.hold_s * args.rate))
    total_s = args.settle_s + args.hold_s

    print(f"joint={name} (policy idx {idx}, driven actuator "
          f"{'rh_' + name[3:5] + 'J0 (coupled)' if idx in (8, 9, 10) else name})")
    print(f"start={start:.4f} rad -> target={target:.4f} rad "
          f"(|step|={abs(target-start):.4f} rad)")
    print(f"rate={args.rate:g} Hz  settle={args.settle_s}s ({n_settle} frames)  "
          f"hold={args.hold_s}s ({n_hold} frames)  total={total_s:.2f}s")
    print(f"expected step frame index (relative to this script's start) = {n_settle}")
    print("Make sure rosbag recording is ALREADY RUNNING with a few seconds of margin "
          f"before/after this script's {total_s:.2f}s ({total_s + 2:.1f}s+ episode duration recommended).")

    pubs = create_hand_publishers()
    rospy.sleep(0.5)  # let publishers connect before the first real command

    rate = rospy.Rate(args.rate)
    cmd = pose.copy()
    cmd[idx] = start

    t_start = rospy.Time.now().to_sec()
    for _ in range(n_settle):
        if rospy.is_shutdown():
            return 1
        publish_to_hand(pubs, cmd)
        rate.sleep()

    cmd[idx] = target
    step_t = rospy.Time.now().to_sec()
    print(f"STEP NOW: t={step_t - t_start:.3f}s since script start")
    for _ in range(n_hold):
        if rospy.is_shutdown():
            return 1
        publish_to_hand(pubs, cmd)
        rate.sleep()

    print("done. Stop recording now (or a couple seconds after) if it hasn't already ended.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except KeyboardInterrupt:
        sys.exit(0)
