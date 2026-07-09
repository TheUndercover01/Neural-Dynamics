"""Thin ROS publisher for excitation trajectories. Publishes raw actuator-space radians
(std_msgs/Float64) to each /sh_rh_<act>_position_controller/command topic — no policy-
space 2x multiplier (see compose.py's module docstring / the plan's "Corrections").

rospy/message imports are INSIDE every function, never at module top level, so the rest
of this package (generators.py, compose.py, test_generators.py) stays importable and
testable on a machine with no ROS install — matching config_lib.py / parse_bag.py's
existing convention (see CLAUDE.md).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402
import config as ecfg  # noqa: E402


def create_publishers() -> dict:
    """One rospy.Publisher per actuator, keyed by actuator name (e.g. 'rh_FFJ0')."""
    import rospy
    from std_msgs.msg import Float64

    joints = cl.load_joints()
    pubs = {}
    for a in joints["actuator_order"]:
        topic = cl.controller_command_topic(a)
        pubs[a] = rospy.Publisher(topic, Float64, queue_size=1)
        rospy.loginfo(f"Publisher: {topic}")
    return pubs


def publish_frame(pubs: dict, frame: np.ndarray, actuator_order: list[str] | None = None) -> None:
    """Publish one (13,) radian frame, raw, in actuator_order — no scaling."""
    from std_msgs.msg import Float64

    actuator_order = actuator_order or cl.load_joints()["actuator_order"]
    for a, val in zip(actuator_order, frame):
        msg = Float64()
        msg.data = float(val)
        pubs[a].publish(msg)


def read_current_actuator_pose(timeout_s: float = 5.0) -> np.ndarray:
    """(13,) radians, actuator_order, from each controller's live process_value. RAW —
    no /2.0 (that division in send_step.py only inverts a policy-space 2x multiplier
    this package never applies; the J0 process_value here already IS the actuator
    command range, 0..pi)."""
    import rospy
    from control_msgs.msg import JointControllerState
    from threading import Lock

    joints = cl.load_joints()
    acts = list(joints["actuator_order"])
    lock = Lock()
    latest: dict[str, float] = {}

    def _mk_cb(name):
        def _cb(msg):
            with lock:
                latest[name] = msg.process_value
        return _cb

    subs = [rospy.Subscriber(cl.controller_state_topic(a), JointControllerState, _mk_cb(a),
                              queue_size=5) for a in acts]

    deadline = rospy.Time.now().to_sec() + timeout_s
    rate = rospy.Rate(20)
    while rospy.Time.now().to_sec() < deadline and not rospy.is_shutdown():
        with lock:
            if len(latest) == len(acts):
                break
        rate.sleep()
    for s in subs:
        s.unregister()

    with lock:
        missing = [a for a in acts if a not in latest]
        if missing:
            raise RuntimeError(f"timed out waiting for current pose from: {missing} "
                                f"(controller /state topics not publishing?)")
        pose = np.array([latest[a] for a in acts], dtype=np.float64)
    return pose


def play(pubs: dict, traj: np.ndarray, rate: float, *, lead_in_ramp_from: np.ndarray | None = None,
         lead_in_s: float | None = None) -> dict:
    """Publish `traj` (T,13) at `rate` Hz with drift-corrected absolute-time scheduling.

    Each frame k is scheduled against t0 + k*dt (t0 = loop start), NEVER against an
    incremental rospy.Rate.sleep() — incremental sleeping accumulates scheduling error
    frame over frame; anchoring to an absolute origin does not.

    If lead_in_ramp_from is given, a linear ramp from that pose to traj[0] over
    lead_in_s (default excitation.config.LEAD_IN_S) is prepended, so an episode never
    opens with a large jump (unsafe, and would itself register as a spurious step).

    Returns {"jitter_ms": {mean,p50,p95,max}, "overruns": int, "n_frames": int} — the
    jitter/overrun stats a caller writes into meta.yaml. If jitter is bad the whole
    dataset built from this episode is suspect.
    """
    import rospy

    joints = cl.load_joints()
    acts = list(joints["actuator_order"])
    dt = 1.0 / rate

    full_traj = traj
    if lead_in_ramp_from is not None:
        lead_in_s = lead_in_s if lead_in_s is not None else ecfg.LEAD_IN_S
        n_ramp = max(1, int(round(lead_in_s * rate)))
        ramp = np.linspace(lead_in_ramp_from, traj[0], n_ramp, endpoint=False)
        full_traj = np.concatenate([ramp, traj], axis=0)

    T = full_traj.shape[0]
    err_ms = np.empty(T, dtype=float)
    overruns = 0

    t0 = rospy.Time.now().to_sec()
    for k in range(T):
        if rospy.is_shutdown():
            err_ms = err_ms[:k]
            break
        target = t0 + k * dt
        now = rospy.Time.now().to_sec()
        err_ms[k] = (now - target) * 1000.0
        if now > target:
            overruns += 1
        publish_frame(pubs, full_traj[k], acts)
        next_target = t0 + (k + 1) * dt
        sleep_s = next_target - rospy.Time.now().to_sec()
        if sleep_s > 0:
            rospy.sleep(sleep_s)

    n = err_ms.size
    jitter_ms = ({"mean": float(err_ms.mean()), "p50": float(np.percentile(err_ms, 50)),
                  "p95": float(np.percentile(err_ms, 95)), "max": float(err_ms.max())}
                 if n else {"mean": None, "p50": None, "p95": None, "max": None})
    return {"jitter_ms": jitter_ms, "overruns": int(overruns), "n_frames": int(n)}


def hold(pubs: dict, pose: np.ndarray, duration_s: float, rate: float | None = None) -> dict:
    """Publish a constant `pose` for duration_s (used for the lead-out hold)."""
    rate = rate or ecfg.PUBLISH_RATE_HZ
    T = max(1, int(round(duration_s * rate)))
    traj = np.tile(np.asarray(pose, dtype=float), (T, 1))
    return play(pubs, traj, rate)
