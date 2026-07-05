#!/usr/bin/env python3
"""Pre-flight stream monitor. Run on the box BEFORE recording an episode.

Confirms all 14 topics publish at the expected rate, prints per-actuator
set_point vs process_value (catches a dead/limp controller), and live-asserts the J0
coupling (FFJ0.process_value == FFJ1 + FFJ2 from /joint_states) as a wiring check.

    rosrun ... check_stream.py            # 5 s sample
    DURATION=10 check_stream.py

Exits non-zero if any topic is silent or the coupling check fails, so it can gate a run.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

import rospy  # noqa: E402
from control_msgs.msg import JointControllerState  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

DURATION = float(os.environ.get("DURATION", "5"))
COUPLING_TOL = float(os.environ.get("COUPLING_TOL", "0.05"))  # rad


def main() -> int:
    joints = cl.load_joints()
    topics = cl.load_topics()
    acts = joints["actuator_order"]
    coupling = cl.coupled_actuators(joints)
    expect = topics.get("expected_rates_hz", {})

    rospy.init_node("actuator_check_stream", anonymous=True, disable_signals=True)

    counts: dict[str, int] = defaultdict(int)
    latest_state: dict[str, JointControllerState] = {}
    latest_js = {"msg": None}

    def on_js(msg: JointState):
        counts["/joint_states"] += 1
        latest_js["msg"] = msg

    subs = [rospy.Subscriber(topics["joint_states"], JointState, on_js, queue_size=50)]
    for a in acts:
        topic = cl.controller_state_topic(a)

        def on_state(msg, _t=topic):
            counts[_t] += 1
            latest_state[_t] = msg

        subs.append(rospy.Subscriber(topic, JointControllerState, on_state, queue_size=50))

    print(f"Sampling {DURATION:.0f}s ...")
    t0 = time.time()
    while time.time() - t0 < DURATION and not rospy.is_shutdown():
        time.sleep(0.05)
    elapsed = time.time() - t0
    for s in subs:
        s.unregister()

    ok = True

    # --- rate report -------------------------------------------------------------------
    print("\n== rates ==")
    js_hz = counts["/joint_states"] / elapsed
    print(f"  {'/joint_states':48s} {js_hz:7.1f} Hz  (expect ~{expect.get('joint_states','?')})")
    if counts["/joint_states"] == 0:
        print("  !! /joint_states SILENT"); ok = False
    for a in acts:
        topic = cl.controller_state_topic(a)
        hz = counts[topic] / elapsed
        flag = "" if counts[topic] else "  !! SILENT"
        if not counts[topic]:
            ok = False
        print(f"  {topic:48s} {hz:7.1f} Hz{flag}")

    # --- per-actuator set_point vs process_value --------------------------------------
    print("\n== set_point vs process_value (rad) ==")
    print(f"  {'actuator':10s} {'set_point':>10s} {'proc_val':>10s} {'error':>10s}")
    for a in acts:
        st = latest_state.get(cl.controller_state_topic(a))
        if st is None:
            print(f"  {a:10s} {'--':>10s} {'--':>10s} {'--':>10s}"); continue
        print(f"  {a:10s} {st.set_point:10.4f} {st.process_value:10.4f} {st.error:10.4f}")

    # --- coupling wiring check ---------------------------------------------------------
    print("\n== J0 coupling check (process_value vs J1+J2) ==")
    js = latest_js["msg"]
    if js is None:
        print("  no /joint_states sample; skipping"); ok = False
    else:
        jidx = {n: i for i, n in enumerate(js.name)}
        for act, (j1, j2) in coupling.items():
            st = latest_state.get(cl.controller_state_topic(act))
            if st is None or j1 not in jidx or j2 not in jidx:
                print(f"  {act}: missing data"); ok = False; continue
            summed = js.position[jidx[j1]] + js.position[jidx[j2]]
            d = abs(st.process_value - summed)
            status = "OK" if d <= COUPLING_TOL else "MISMATCH"
            if d > COUPLING_TOL:
                ok = False
            print(f"  {act}: process_value={st.process_value:.4f}  "
                  f"{j1}+{j2}={summed:.4f}  |d|={d:.4f}  {status}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
