#!/usr/bin/env python3
"""Extract per-topic time series from a rosbag, keyed by header stamp.

Library:  parse_bag(path) -> dict of numpy arrays (see below).
CLI:      parse_bag.py BAG [BAG ...]   # prints a drop/jitter QC report per bag.

Returned structure:
    {
      "joint_states": {"t":[N], "position":[N,16], "velocity":[N,16], "effort":[N,16],
                        "name":[16 str]},
      "controller":   {actuator: {"t":[M], "set_point":[M], "process_value":[M],
                                  "process_value_dot":[M], "error":[M], "command":[M]}},
      "report": {...jitter/drop stats...},
    }
The bag is the immutable source of truth; this module only reads it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402


def _stamps_report(t: np.ndarray, expected_hz: float) -> dict:
    """Rate/jitter/drop stats for a monotonic-ish stamp series."""
    if t.size < 2:
        return {"n": int(t.size), "rate_hz": 0.0, "non_monotonic": 0, "gaps": 0}
    dt = np.diff(t)
    non_mono = int(np.sum(dt <= 0))
    period = 1.0 / expected_hz if expected_hz else float(np.median(dt[dt > 0]) or 0.0)
    gaps = int(np.sum(dt > 2.5 * period)) if period else 0
    good = dt[dt > 0]
    return {
        "n": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "rate_hz": float(good.size / (t[-1] - t[0])) if t[-1] > t[0] else 0.0,
        "dt_median_ms": float(np.median(good) * 1e3) if good.size else 0.0,
        "dt_p99_ms": float(np.percentile(good, 99) * 1e3) if good.size else 0.0,
        "non_monotonic": non_mono,
        "gaps": gaps,
    }


def parse_bag(path: str | pathlib.Path) -> dict:
    import rosbag  # lazy: only bag reading needs ROS; align/build/normalize do not

    joints = cl.load_joints()
    topics = cl.load_topics()
    acts = joints["actuator_order"]
    joint_order = joints["joint_order"]
    expect = topics.get("expected_rates_hz", {})

    js_topic = topics["joint_states"]
    state_topic_by_act = {a: cl.controller_state_topic(a) for a in acts}
    wanted = [js_topic] + list(state_topic_by_act.values())

    js_t, js_pos, js_vel, js_eff = [], [], [], []
    js_name: list[str] | None = None
    ctrl: dict[str, dict[str, list]] = {
        a: {k: [] for k in ("t", "set_point", "process_value",
                             "process_value_dot", "error", "command")}
        for a in acts}
    topic_to_act = {v: k for k, v in state_topic_by_act.items()}

    with rosbag.Bag(str(path)) as bag:
        for topic, msg, _ in bag.read_messages(topics=wanted):
            if topic == js_topic:
                if js_name is None:
                    js_name = list(msg.name)
                    assert js_name == joint_order, (
                        "joint_states name order in bag does not match joints.yaml:\n"
                        f"  bag: {js_name}\n  cfg: {joint_order}")
                js_t.append(msg.header.stamp.to_sec())
                js_pos.append(msg.position)
                js_vel.append(msg.velocity)
                js_eff.append(msg.effort)
            else:
                a = topic_to_act[topic]
                c = ctrl[a]
                c["t"].append(msg.header.stamp.to_sec())
                c["set_point"].append(msg.set_point)
                c["process_value"].append(msg.process_value)
                c["process_value_dot"].append(msg.process_value_dot)
                c["error"].append(msg.error)
                c["command"].append(msg.command)

    if js_name is None:
        raise RuntimeError(f"{path}: no {js_topic} messages found")

    out = {
        "joint_states": {
            "t": np.asarray(js_t, float),
            "position": np.asarray(js_pos, float),
            "velocity": np.asarray(js_vel, float),
            "effort": np.asarray(js_eff, float),
            "name": js_name,
        },
        "controller": {},
        "report": {"joint_states": _stamps_report(
            np.asarray(js_t, float), expect.get("joint_states", 0))},
    }
    for a in acts:
        c = {k: np.asarray(v, float) for k, v in ctrl[a].items()}
        out["controller"][a] = c
        out["report"][a] = _stamps_report(c["t"], expect.get("controller_state", 0))
        if c["t"].size == 0:
            out["report"][a]["MISSING"] = True
    return out


def _print_report(path: str, parsed: dict) -> None:
    print(f"\n=== {path} ===")
    r = parsed["report"]
    print(f"  {'stream':28s} {'n':>7s} {'rate':>7s} {'dt_med':>8s} {'dt_p99':>8s} "
          f"{'nonmono':>8s} {'gaps':>5s}")
    for name in ["joint_states"] + list(parsed["controller"].keys()):
        s = r[name]
        miss = "  <-- MISSING" if s.get("MISSING") else ""
        print(f"  {name:28s} {s['n']:7d} {s['rate_hz']:7.1f} "
              f"{s.get('dt_median_ms',0):8.2f} {s.get('dt_p99_ms',0):8.2f} "
              f"{s['non_monotonic']:8d} {s['gaps']:5d}{miss}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        _print_report(p, parse_bag(p))
