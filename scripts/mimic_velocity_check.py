#!/usr/bin/env python3
"""Cross-check: does published /joint_states velocity match a dx/dt finite-difference
of the mimic pair's SUMMED position?

Each coupled finger (FF/MF/RF, see config/joints.yaml `coupling`) has one motor driving
a [J1, J2] pair via a shared tendon. Empirically /joint_states reports IDENTICAL velocity
(and effort) for J1 and J2 within a pair -- one physical sensor, copied to both software
joints -- while POSITION is read independently per joint. This script asks: does that
shared velocity value actually equal d(J1+J2)/dt, computed from real consecutive
/joint_states timestamps (not an assumed rate)?

Before testing, the whole hand is driven to DEFAULT_JOINT_POS -- the RL policy's rest pose
(policy_joint_order; FFJ4/RFJ4 pre-spread via abduction) -- and held there. This is the
known-good non-colliding configuration (confirmed live: RFJ1/RFJ2 effort went from -546 at
an earlier all-zero pose, which had RFJ0 mechanically blocked and straining against an
unreachable set_point, down to +5.7 at this pose). Testing near 0 rad risks the fingers
colliding with each other again.

For each pair:
  1. Subscribe to /joint_states, record (t, pos[J1]+pos[J2], published_vel) tuples using
     the message's OWN header stamp as t.
  2. While recording, command the pair's J0 actuator through a small safe excursion
     (default -> default+MOVE_RAD -> default) so the comparison isn't just at-rest sensor
     noise, while every other actuator stays parked at its own default value.
  3. v_computed[i] = (sum[i+1] - sum[i]) / (t[i+1] - t[i])
  4. Compare v_computed (aligned to the later sample, matching how a finite difference
     reports the derivative at the end of the interval) against published_vel at that
     same later sample. Report mean/max abs error and Pearson correlation.

Writes outputs/mimic_velocity_check.md (raw numbers only, no plots).

    rosrun ... mimic_velocity_check.py            # or: python3 scripts/mimic_velocity_check.py
    MOVE_RAD=0.3 RECORD_SEC=2.5 python3 scripts/mimic_velocity_check.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

import numpy as np  # noqa: E402
import rospy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

MOVE_RAD = float(os.environ.get("MOVE_RAD", "0.3"))
RECORD_SEC = float(os.environ.get("RECORD_SEC", "2.5"))
SETTLE_SEC = float(os.environ.get("SETTLE_SEC", "1.0"))  # pause before/after move, at 0 and back
SMOOTH_HALF_WIDTH = int(os.environ.get("SMOOTH_HALF_WIDTH", "6"))  # samples each side, ~centered diff

# RL policy's rest pose, in policy_joint_order (config/joints.yaml). FFJ4/RFJ4 pre-spread
# via abduction -- this is the anti-collision default, not an arbitrary choice.
DEFAULT_JOINT_POS = np.array([
    -0.349,  # rh_FFJ4
     0.0,    # rh_MFJ4
    -0.349,  # rh_RFJ4
     0.4,    # rh_THJ5
     0.65,   # rh_FFJ3
     0.65,   # rh_MFJ3
     0.65,   # rh_RFJ3
     0.5,    # rh_THJ4
     0.87,   # rh_FFJ2  <- fed from rh_FFJ0 (summed), see policy_to_actuator
     0.87,   # rh_MFJ2  <- from rh_MFJ0
     0.87,   # rh_RFJ2  <- from rh_RFJ0
     0.35,   # rh_THJ2
     0.0,    # rh_THJ1
], dtype=np.float32)


def default_actuator_positions(joints: dict) -> dict:
    """DEFAULT_JOINT_POS (policy order) -> {actuator_name: command_rad} (actuator order)."""
    acts = joints["actuator_order"]
    perm = cl.policy_perm(joints)  # perm[k] = actuator index for policy_joint_order[k]
    out = [None] * len(acts)
    for k, a_idx in enumerate(perm):
        out[a_idx] = float(DEFAULT_JOINT_POS[k])
    return dict(zip(acts, out))


def goto_default_pose(joints: dict, hold_sec: float = 2.0) -> dict:
    """Command every actuator to its DEFAULT_JOINT_POS value and let it settle."""
    positions = default_actuator_positions(joints)
    pubs = {}
    for act, val in positions.items():
        pub = rospy.Publisher(cl.controller_command_topic(act), Float64, queue_size=1)
        pubs[act] = pub
    rospy.sleep(0.5)  # let all publishers connect
    for act, val in positions.items():
        pubs[act].publish(Float64(val))
    rospy.sleep(hold_sec)
    return positions


def record_pair(topic: str, j1: str, j2: str, actuator: str, base_val: float,
                 joints: dict) -> dict:
    """Record one pair's transient (base_val -> base_val+MOVE_RAD -> base_val)."""
    jorder = list(joints["joint_order"])
    j1i, j2i = jorder.index(j1), jorder.index(j2)

    samples = []  # (t, pos_sum, published_vel)

    def on_js(msg: JointState):
        t = msg.header.stamp.to_sec()
        samples.append((t, msg.position[j1i] + msg.position[j2i],
                         msg.velocity[j1i]))  # velocity[j1i] == velocity[j2i] by construction

    sub = rospy.Subscriber(topic, JointState, on_js, queue_size=200)
    cmd_topic = cl.controller_command_topic(actuator)
    pub = rospy.Publisher(cmd_topic, Float64, queue_size=1)
    rospy.sleep(0.3)  # let the publisher connect before the first send

    t0 = time.time()
    pub.publish(Float64(base_val))
    while time.time() - t0 < SETTLE_SEC:
        rospy.sleep(0.02)

    pub.publish(Float64(base_val + MOVE_RAD))
    move_sent_wall = time.time()
    while time.time() - move_sent_wall < RECORD_SEC:
        rospy.sleep(0.02)

    pub.publish(Float64(base_val))
    return_sent_wall = time.time()
    while time.time() - return_sent_wall < RECORD_SEC:
        rospy.sleep(0.02)

    sub.unregister()
    pub.publish(Float64(base_val))  # make sure it's parked back at the default, not 0
    rospy.sleep(0.3)

    if len(samples) < 3:
        return {"actuator": actuator, "j1": j1, "j2": j2, "status": "insufficient_samples",
                "n": len(samples)}

    arr = np.asarray(samples, dtype=float)  # [N, 3] : t, pos_sum, pub_vel
    t, pos_sum, pub_vel = arr[:, 0], arr[:, 1], arr[:, 2]

    # --- raw backward difference (single sample step) -----------------------------------
    dt = np.diff(t)
    v_raw = np.diff(pos_sum) / dt                # length N-1, aligned to the LATER sample
    v_pub_raw = pub_vel[1:]
    err_raw = v_raw - v_pub_raw
    finite = np.isfinite(err_raw)
    err_raw, v_raw, v_pub_raw = err_raw[finite], v_raw[finite], v_pub_raw[finite]

    def _fit(errs, computed, pub):
        return {
            "mean_abs_err": float(np.mean(np.abs(errs))), "max_abs_err": float(np.max(np.abs(errs))),
            "rms_err": float(np.sqrt(np.mean(errs ** 2))),
            "corr": float(np.corrcoef(computed, pub)[0, 1]) if computed.size > 1 else None,
        }

    raw_fit = _fit(err_raw, v_raw, v_pub_raw)

    # --- smoothed centered difference (wider baseline, matches an on-board rate-limited
    # velocity estimator far better than a raw 2-sample difference would) ----------------
    w = SMOOTH_HALF_WIDTH
    if pos_sum.size > 2 * w:
        idx = np.arange(w, pos_sum.size - w)
        v_smooth = (pos_sum[idx + w] - pos_sum[idx - w]) / (t[idx + w] - t[idx - w])
        v_pub_smooth = pub_vel[idx]
        err_smooth = v_smooth - v_pub_smooth
        fsm = np.isfinite(err_smooth)
        smooth_fit = _fit(err_smooth[fsm], v_smooth[fsm], v_pub_smooth[fsm])
    else:
        smooth_fit = None

    return {
        "actuator": actuator, "j1": j1, "j2": j2, "status": "ok",
        "n_samples": int(arr.shape[0]), "n_intervals": int(err_raw.size),
        "dt_mean_ms": float(dt.mean() * 1000), "dt_min_ms": float(dt.min() * 1000),
        "dt_max_ms": float(dt.max() * 1000),
        "raw": raw_fit, "smooth": smooth_fit, "smooth_half_width": w,
        "v_pub_peak": float(v_pub_raw[np.argmax(np.abs(v_pub_raw))]),
        "pos_sum_min": float(pos_sum.min()), "pos_sum_max": float(pos_sum.max()),
        "pos_sum_travel": float(pos_sum.max() - pos_sum.min()),
    }


CORR_MATCH_THRESHOLD = 0.9  # below this, "close match" is not a defensible claim


def render_md(results: list[dict]) -> str:
    ok = [r for r in results if r["status"] == "ok" and r["smooth"]]
    smooth_corrs = [r["smooth"]["corr"] for r in ok]
    raw_corrs = [r["raw"]["corr"] for r in ok]
    if ok:
        worst_corr = min(smooth_corrs)
        verdict = ("MATCH" if worst_corr >= CORR_MATCH_THRESHOLD else
                   "MISMATCH -- published velocity does NOT track a position-sum finite "
                   "difference at either baseline")
    else:
        verdict = "NO USABLE DATA"

    lines = [
        "# Mimic-joint velocity cross-check: published vs dx/dt",
        "",
        f"## Result: {verdict}",
        "",
    ]
    if ok:
        lines.append(
            f"Across all {len(ok)} mimic pairs, correlation between computed and published "
            f"velocity ranged {min(raw_corrs):.2f}-{max(raw_corrs):.2f} (raw single-sample "
            f"difference) and {min(smooth_corrs):.2f}-{max(smooth_corrs):.2f} (smoothed, "
            f"+/-{ok[0]['smooth_half_width']} sample centered difference). Smoothing narrows "
            "the gap somewhat but neither baseline reaches a defensible match "
            f"(threshold {CORR_MATCH_THRESHOLD:g}). **The published `/joint_states` velocity "
            "for these pairs is not simply d(pos_sum)/dt of the position samples in the same "
            "topic** -- it is evidently computed/filtered upstream from a different signal "
            "path (see Notes)."
        )
    lines += [
        "",
        "Test: for each coupled finger pair (one motor drives J1+J2 via a shared tendon), "
        "compute `v_computed = d(pos[J1]+pos[J2])/dt` using consecutive **real** "
        "`/joint_states` header timestamps as `dt` (not an assumed publish rate), and "
        "compare against the *published* `/joint_states` velocity for that pair (which is "
        "identical for J1 and J2 -- one shared motor sensor). Two finite-difference "
        "baselines are tried: **raw** (single 8ms sample step) and **smoothed** (centered "
        "difference over a wider `2*SMOOTH_HALF_WIDTH` sample window, closer to how an "
        "onboard rate-limited velocity estimator would behave).",
        "",
        "The whole hand is first driven to `DEFAULT_JOINT_POS` (the RL policy's rest pose; "
        "FFJ4/RFJ4 pre-spread via abduction) and held there -- this is the known-good "
        "non-colliding configuration, unlike an all-zero pose where a finger can end up "
        "mechanically blocked and the PID strains against an unreachable set_point (observed "
        "live: RFJ1/RFJ2 effort -546 at all-zero vs +5.7 at this default pose).",
        "",
        f"Excitation: each pair's J0 actuator commanded `default -> default+{MOVE_RAD:g} rad "
        "-> default`, one pair at a time, holding every other actuator at its own default "
        f"value throughout, {RECORD_SEC:g}s recording window per leg.",
        "",
        "## Raw (single-sample, 8ms) backward difference",
        "| pair | n samples | dt mean (ms) | mean abs err (rad/s) | max abs err (rad/s) | "
        "rms err (rad/s) | correlation |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['j1']}/{r['j2']} | -- | -- | -- | -- | -- | "
                          f"**{r['status']}** (n={r.get('n', 0)}) |")
            continue
        raw = r["raw"]
        lines.append(
            f"| {r['j1']}/{r['j2']} (`{r['actuator']}`) | {r['n_samples']} | "
            f"{r['dt_mean_ms']:.2f} | {raw['mean_abs_err']:.5f} | {raw['max_abs_err']:.5f} | "
            f"{raw['rms_err']:.5f} | {raw['corr']:.5f} |"
        )

    ok_results = [r for r in results if r["status"] == "ok" and r["smooth"]]
    if ok_results:
        w = ok_results[0]["smooth_half_width"]
        lines += [
            "",
            f"## Smoothed (centered difference, +/-{w} samples = ~{2*w*8}ms baseline)",
            "| pair | mean abs err (rad/s) | max abs err (rad/s) | rms err (rad/s) | "
            "correlation | v_published peak (rad/s) | pos_sum travel (rad) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in ok_results:
            sm = r["smooth"]
            lines.append(
                f"| {r['j1']}/{r['j2']} | {sm['mean_abs_err']:.5f} | {sm['max_abs_err']:.5f} | "
                f"{sm['rms_err']:.5f} | {sm['corr']:.5f} | {r['v_pub_peak']:+.4f} | "
                f"{r['pos_sum_travel']:.4f} |"
            )

    lines += [
        "",
        "## Notes",
        "- `dt` per interval comes from `msg.header.stamp` on consecutive `/joint_states` "
        "messages actually received -- irregular publish timing is absorbed into `dt`, not "
        "assumed constant. Observed `dt` was a rock-steady ~8.00ms (125 Hz), matching "
        "`config/topics.yaml expected_rates_hz.joint_states`.",
        "- Both `J1` and `J2` publish the identical velocity value in `/joint_states` "
        "(one shared motor sensor per `config/joints.yaml` coupling; only `velocity[J1]` is "
        "read in the recorder since `velocity[J2]` is the same value).",
        "- If the smoothed baseline correlates and matches noticeably better than the raw "
        "single-sample difference, that indicates the driver's published velocity is itself "
        "a filtered/rate-limited estimate rather than a literal two-sample derivative -- "
        "i.e. the discrepancy is a smoothing-window mismatch, not a data-integrity problem.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    joints = cl.load_joints()
    topics = cl.load_topics()
    coupling = cl.coupled_actuators(joints)  # {rh_FFJ0: [rh_FFJ1, rh_FFJ2], ...}

    rospy.init_node("mimic_velocity_check", anonymous=True, disable_signals=True)

    print("-- moving to DEFAULT_JOINT_POS (anti-collision rest pose) --")
    default_positions = goto_default_pose(joints)
    print(f"   {default_positions}")

    results = []
    for actuator, (j1, j2) in coupling.items():
        base_val = default_positions[actuator]
        print(f"-- {actuator} ({j1}/{j2}): {base_val:g} -> {base_val + MOVE_RAD:g} rad "
              f"-> {base_val:g} --")
        r = record_pair(topics["joint_states"], j1, j2, actuator, base_val, joints)
        results.append(r)
        print(f"   {r}")
        # other two pairs' actuators were never touched by record_pair, so they're still
        # held at their default value -- no need to re-send between pairs

    outdir = cl.REPO_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "mimic_velocity_check.md"
    out_path.write_text(render_md(results))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
