#!/usr/bin/env python3
"""Measure per-joint actuation/sensing DELAY and TIME CONSTANT/settling from
step-response data, and recommend the actuator-net history-window floor.

    latency_timeconstant.py --data data/aligned/**/*.aligned.npz
    latency_timeconstant.py --data data/raw/2026_07_02_am/ep001_*.bag
    latency_timeconstant.py --live      # optional, hardware-only, not implemented here

Consumes this repo's own aligned-table schema (preprocess/align.py output): t,
act_pos[T,13] (controller process_value, actuator order), action[T,13] (set_point),
gt_pos[T,16] (joint_states position, joint order), seg_id (contiguous-valid-run id,
-1 = gap). A raw .bag is also accepted; it is aligned in-memory via align_parsed()
before analysis (no file is written).

Concepts (see Dactyl/PDDM):
  DELAY     = frames between a command jump and the joint FIRST responding.
              Sets the minimum history the net needs to see the cause.
  TIME CONSTANT / SETTLING = frames from motion-start to ~63% / ~95% of the step.
              Spans the transient the net must observe.
  WINDOW FLOOR = max_over_joints(delay + settling), worst case, not the mean.

Every one of the 16 /joint_states joints is covered: the 3 J0 actuators
(FFJ0/MFJ0/RFJ0) each drive a coupled [J1, J2] pair (see config/joints.yaml
`coupling`) and are measured on BOTH driven joints against the one shared command;
the other 10 actuators map 1:1 to the identically-named joint. Both the
controller-side (act_pos) and ground-truth (gt_pos) response are measured per step;
their difference is the tendon-compliance signal.

Writes outputs/latency_report.json, outputs/step_response_<joint>.png (one per
joint with at least one usable step), outputs/memory_summary.png, and prints a
plain-English summary + verdict line.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config_lib as cl  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

# dataviz palette (references/palette.md): fixed categorical order, muted ink for
# non-data chrome. Only two real series appear (command vs measured) so this stays
# well inside the validated set; markers/thresholds use muted ink, not new hues.
_BLUE = "#2a78d6"      # command / delay segment
_AQUA = "#1baf7a"      # measured / settling segment
_RED = "#e34948"        # worst-case highlight (status: critical family)
_MUTED = "#898781"      # marker lines / reference lines
_GRID = "#e1e0d9"
_INK = "#0b0b0b"


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def _load_one(path: pathlib.Path) -> dict:
    if path.suffix == ".bag":
        from preprocess.parse_bag import parse_bag
        from preprocess.align import align_parsed
        parsed = parse_bag(path)
        return align_parsed(parsed, cl.load_pipeline(), cl.load_joints())
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_tag(path: pathlib.Path) -> str | None:
    """Best-effort: aligned.npz -> meta/<session>/<stem>.json -> 'excitation' field.
    Returns None if the sidecar is missing (no data-model assumption is asserted; this
    is purely optional enrichment)."""
    try:
        session = path.parent.name
        stem = path.name
        if stem.endswith(".aligned.npz"):
            stem = stem[: -len(".aligned.npz")]
        sidecar = cl.REPO_ROOT / "meta" / session / f"{stem}.json"
        if not sidecar.exists():
            return None
        return json.loads(sidecar.read_text()).get("excitation")
    except Exception:  # noqa: BLE001
        return None


def load_episodes(data_args: list[str]) -> list[dict]:
    """Resolve --data globs/paths (aligned .npz and/or raw .bag) to episode dicts."""
    paths: list[pathlib.Path] = []
    for a in data_args:
        matches = sorted(pathlib.Path(p) for p in glob.glob(a, recursive=True))
        paths.extend(matches if matches else [pathlib.Path(a)])
    episodes = []
    for p in paths:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        ep = _load_one(p)
        ep["path"] = p
        ep["load_tag"] = _load_tag(p) if p.suffix == ".npz" else None
        episodes.append(ep)
    return episodes


def check_rate(episodes: list[dict], expected_rate: float) -> float:
    rates = sorted({float(ep["dataset_rate"]) for ep in episodes})
    if len(rates) > 1:
        print(f"WARNING: episodes were built at different rates: {rates}Hz — "
              f"frame counts below are not comparable across episodes.", file=sys.stderr)
    observed = rates[0] if rates else expected_rate
    if abs(observed - expected_rate) > 1e-6:
        pipeline_rate = cl.load_pipeline().get("dataset_rate")
        print(f"WARNING: data was resampled at {observed:g} Hz, not the "
              f"{expected_rate:g} Hz this diagnostic's window-floor comparison assumes "
              f"(config/pipeline.yaml currently has dataset_rate={pipeline_rate}). "
              f"Frame<->ms math below uses the data's OWN rate ({observed:g} Hz); only "
              f"the raw current_obs_frames/current_action_frames comparison assumes 60Hz "
              f"frames — re-derive those if this hand's actuator net actually runs at "
              f"{observed:g} Hz.", file=sys.stderr)
    return observed


# --------------------------------------------------------------------------------------
# Step detection (per command channel: 13 actuators' `action` / set_point trace)
# --------------------------------------------------------------------------------------

def detect_steps(action: np.ndarray, seg_id: np.ndarray, step_thresh: float,
                  hold_frames: int) -> list[dict]:
    """Command jumps by > step_thresh rad and then holds (within step_thresh/4) for
    >= hold_frames, all inside one contiguous seg_id run (never across a data gap)."""
    T = action.size
    steps = []
    k = 1
    while k < T:
        if seg_id[k] < 0 or seg_id[k - 1] != seg_id[k]:
            k += 1
            continue
        delta = action[k] - action[k - 1]
        if abs(delta) > step_thresh:
            end = k + hold_frames
            if end >= T or np.any(seg_id[k:end] != seg_id[k]):
                k += 1
                continue
            if np.max(np.abs(action[k:end] - action[k])) > step_thresh * 0.25:
                k += 1
                continue
            steps.append({"step_frame": k, "pre_value": float(action[k - 1]),
                          "post_value": float(action[k]), "magnitude": float(delta)})
            k = end  # don't re-trigger inside the hold window just consumed
            continue
        k += 1
    return steps


# --------------------------------------------------------------------------------------
# Per-step response measurement (delay, tau, settling) on one measured trace
# --------------------------------------------------------------------------------------

def measure_response(measured: np.ndarray, seg_id: np.ndarray, step_frame: int,
                      magnitude: float, response_window: int, move_thresh: float) -> dict:
    T = measured.size
    end = step_frame + response_window
    pre_n = min(5, step_frame)
    if end > T or pre_n == 0 or np.any(seg_id[step_frame:end] != seg_id[step_frame]):
        return {"status": "boundary_excluded"}

    pre_value = float(np.median(measured[step_frame - pre_n:step_frame]))
    window = measured[step_frame:end]
    thresh = move_thresh * abs(magnitude)
    moved = np.abs(window - pre_value) > thresh
    if not moved.any():
        return {"status": "no_response", "pre_value": pre_value}

    motion_start = step_frame + int(np.argmax(moved))
    delay_frames = motion_start - step_frame
    final_settled = float(np.median(measured[end - 5:end]))
    total_travel = final_settled - pre_value
    direction = np.sign(total_travel) if total_travel != 0 else np.sign(magnitude)

    # The window may close before the signal has actually stopped moving (long tau
    # relative to response_window): "final_settled" would then be a point mid-transient,
    # biasing total_travel low and everything derived from it. Flag it rather than
    # silently reporting an underestimate.
    window_limited = bool(total_travel) and abs(measured[end - 1] - measured[end - 5]) > \
        0.05 * abs(total_travel)

    def _cross(frac: float) -> int | None:
        target = pre_value + frac * total_travel
        sub = measured[motion_start:end]
        hit = np.where(sub >= target)[0] if direction >= 0 else np.where(sub <= target)[0]
        return motion_start + int(hit[0]) if hit.size else None

    t63, t95 = _cross(0.63), _cross(0.95)
    status = "ok" if (t63 is not None and t95 is not None) else "not_fully_settled"
    return {
        "status": status, "pre_value": pre_value, "final_settled": final_settled,
        "total_travel": total_travel, "motion_start": motion_start,
        "delay_frames": delay_frames, "magnitude": magnitude,
        "tau_frames": (t63 - motion_start) if t63 is not None else None,
        "settling_frames": (t95 - motion_start) if t95 is not None else None,
        "t63": t63, "t95": t95, "step_frame": step_frame, "window_limited": window_limited,
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median": float(med), "iqr": float(q3 - q1), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def _to_ms(stat: dict | None, rate: float) -> dict | None:
    if stat is None:
        return None
    return {k: (v * 1000.0 / rate if k in ("median", "iqr", "min", "max") else v)
            for k, v in stat.items()}


def _aggregate_side(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] in ("ok", "not_fully_settled")]
    delay = _stats([r["delay_frames"] for r in ok])
    tau = _stats([r["tau_frames"] for r in ok if r["tau_frames"] is not None])
    settling = _stats([r["settling_frames"] for r in ok if r["settling_frames"] is not None])
    worst_case = max(
        (r["delay_frames"] + r["settling_frames"] for r in ok if r["settling_frames"] is not None),
        default=None)

    def _by() -> dict:
        if not ok:
            return {}
        mags = np.abs([r["magnitude"] for r in ok])
        med_mag = float(np.median(mags))
        small = [r for r, m in zip(ok, mags) if m <= med_mag]
        large = [r for r, m in zip(ok, mags) if m > med_mag]
        out = {}
        for label, subset in (("small", small), ("large", large)):
            if subset:
                out[label] = {"n": len(subset), "delay_frames": _stats(
                    [r["delay_frames"] for r in subset])}
        if len(out) < 2:
            out["note"] = "insufficient step-size variation to split small vs large"
        return out

    return {
        "n_steps": len(records),
        "n_ok": len(ok),
        "n_no_response": sum(1 for r in records if r["status"] == "no_response"),
        "n_boundary_excluded": sum(1 for r in records if r["status"] == "boundary_excluded"),
        "n_not_fully_settled": sum(1 for r in records if r["status"] == "not_fully_settled"),
        "n_window_limited": sum(1 for r in ok if r.get("window_limited")),
        "delay_frames": delay, "tau_frames": tau, "settling_frames": settling,
        "jitter_frames": delay["iqr"] if delay else None,
        "worst_case_frames": worst_case,
        "by_size": _by(),
        "records": ok,  # kept for plotting; stripped before json.dump
    }


def build_joint_reports(episodes: list[dict], joints: dict, cfg: dict, rate: float) -> dict:
    acts = list(joints["actuator_order"])
    jorder = list(joints["joint_order"])
    coupling = cl.coupled_actuators(joints)
    jidx = {n: i for i, n in enumerate(jorder)}

    # actuator -> [(joint_name, is_coupled)]
    driven = {}
    for a in acts:
        driven[a] = list(coupling[a]) if a in coupling else [a]

    per_actuator_records: dict[str, list[dict]] = {a: [] for a in acts}
    per_joint_records: dict[str, list[dict]] = {j: [] for j in jorder}
    load_tags_seen: dict[str, set] = {j: set() for j in jorder}

    for ep_idx, ep in enumerate(episodes):
        seg_id = ep["seg_id"]
        for ai, a in enumerate(acts):
            steps = detect_steps(ep["action"][:, ai], seg_id, cfg["step_thresh"],
                                  cfg["hold_frames"])
            for s in steps:
                ctrl_r = measure_response(ep["act_pos"][:, ai], seg_id, s["step_frame"],
                                          s["magnitude"], cfg["response_window"],
                                          cfg["move_thresh"])
                ctrl_r["magnitude"] = s["magnitude"]
                ctrl_r["episode_idx"] = ep_idx
                per_actuator_records[a].append(ctrl_r)
                for j in driven[a]:
                    gt_r = measure_response(ep["gt_pos"][:, jidx[j]], seg_id, s["step_frame"],
                                            s["magnitude"], cfg["response_window"],
                                            cfg["move_thresh"])
                    gt_r["magnitude"] = s["magnitude"]
                    gt_r["episode_idx"] = ep_idx
                    per_joint_records[j].append(gt_r)
                    if ep.get("load_tag"):
                        load_tags_seen[j].add(ep["load_tag"])

    per_joint = {}
    any_load_tags = any(load_tags_seen.values())
    for j in jorder:
        a = next(act for act, js in driven.items() if j in js)
        gt_side = _aggregate_side(per_joint_records[j])
        ctrl_side = _aggregate_side(per_actuator_records[a])

        def _diff(key):
            g, c = gt_side[key], ctrl_side[key]
            return (g["median"] - c["median"]) if (g and c) else None

        gt_recs = gt_side.pop("records")
        ctrl_recs = ctrl_side.pop("records")

        # Start from every field _aggregate_side computed for the gt (joint) side, so a
        # new field added there automatically shows up here without another hand-edit.
        entry = {
            **gt_side,
            "driven_by_actuator": a,
            "delay_ms": _to_ms(gt_side["delay_frames"], rate),
            "tau_ms": _to_ms(gt_side["tau_frames"], rate),
            "settling_ms": _to_ms(gt_side["settling_frames"], rate),
            "by_load": {"available": bool(load_tags_seen[j]),
                        "tags_seen": sorted(load_tags_seen[j])} if any_load_tags else
                       {"available": False,
                        "note": "no excitation/load tag found alongside this data; "
                                "the floor below may be an underestimate if loaded-"
                                "condition settling is slower (see COLLECTION_PROTOCOL.md)"},
            "ctrl_side": {k: v for k, v in ctrl_side.items()},
            "ctrl_vs_gt_diff_frames": {
                "delay": _diff("delay_frames"), "tau": _diff("tau_frames"),
                "settling": _diff("settling_frames"),
            },
            "_gt_records": gt_recs, "_ctrl_records": ctrl_recs,  # for plotting only
        }
        per_joint[j] = entry
    return per_joint


# --------------------------------------------------------------------------------------
# Window-floor recommendation
# --------------------------------------------------------------------------------------

def recommend(per_joint: dict, cfg: dict, rate: float) -> dict:
    worst_by_joint = {}
    for j, e in per_joint.items():
        candidates = [e["worst_case_frames"], e["ctrl_side"].get("worst_case_frames")]
        candidates = [c for c in candidates if c is not None]
        if candidates:
            worst_by_joint[j] = max(candidates)
    delay_by_joint = {j: e["delay_frames"]["median"] + (e["jitter_frames"] or 0)
                      for j, e in per_joint.items() if e["delay_frames"]}

    if not worst_by_joint:
        return {"suggested_floor_frames": None, "suggested_action_obs_offset": None,
                "verdict": "NO USABLE STEPS FOUND — cannot compute a recommendation. "
                           "Collect deliberate step commands (hold, jump, hold ~1s per "
                           "joint) per COLLECTION_PROTOCOL.md and re-run."}

    worst_joint = max(worst_by_joint, key=worst_by_joint.get)
    suggested_floor = int(np.ceil(worst_by_joint[worst_joint]))
    delay_joint = max(delay_by_joint, key=delay_by_joint.get) if delay_by_joint else None
    suggested_offset = int(np.ceil(delay_by_joint[delay_joint])) if delay_joint else None

    obs = cfg["current_obs_frames"]
    act = cfg["current_action_frames"]
    if suggested_floor <= obs and (suggested_offset is None or suggested_offset <= act):
        verdict = f"current window sufficient (obs={obs} >= floor={suggested_floor} frames)"
    elif suggested_offset is not None and suggested_offset > act:
        verdict = (f"extend window to >= {suggested_floor} frames AND increase "
                   f"action/obs offset to >= {suggested_offset} frames (measured "
                   f"worst-case delay at {delay_joint} exceeds current_action_frames={act})")
    else:
        verdict = f"extend window to >= {suggested_floor} frames (current obs={obs})"

    return {
        "suggested_floor_frames": suggested_floor, "worst_case_joint": worst_joint,
        "suggested_action_obs_offset": suggested_offset,
        "delay_driving_joint": delay_joint, "verdict": verdict,
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def _pick_representative(records: list[dict]) -> dict | None:
    ok = [r for r in records if r["status"] in ("ok", "not_fully_settled")]
    return max(ok, key=lambda r: abs(r["magnitude"])) if ok else None


def plot_step_response(joint: str, entry: dict, episodes: list[dict], joints: dict,
                        rate: float, outdir: pathlib.Path) -> pathlib.Path | None:
    rec = _pick_representative(entry["_gt_records"])
    if rec is None:
        return None
    ep = episodes[rec["episode_idx"]]
    acts = list(joints["actuator_order"])
    jorder = list(joints["joint_order"])
    action_trace = ep["action"][:, acts.index(entry["driven_by_actuator"])]
    measured_trace = ep["gt_pos"][:, jorder.index(joint)]

    sf = rec["step_frame"]
    lo = max(0, sf - 5)
    hi = min(len(action_trace), sf + max(30, (rec["t95"] or rec["motion_start"] + 10) - sf + 5))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(lo, hi)
    ax.plot(xs, action_trace[lo:hi], color=_BLUE, lw=2, label="command (set_point)")
    ax.plot(xs, measured_trace[lo:hi], color=_AQUA, lw=2, label="measured (gt_pos)")
    ax.axvline(sf, color=_MUTED, ls="--", lw=1)
    ax.axvline(rec["motion_start"], color=_AQUA, ls="--", lw=1)
    if rec["t63"] is not None:
        ax.axvline(rec["t63"], color=_RED, ls=":", lw=1)
    if rec["t95"] is not None:
        ax.axvline(rec["t95"], color=_INK, ls=":", lw=1)
    ax.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=_GRID, lw=0.6)
    delay_ms = rec["delay_frames"] * 1000.0 / rate
    tau_ms = (rec["tau_frames"] or 0) * 1000.0 / rate
    settle_ms = (rec["settling_frames"] or 0) * 1000.0 / rate
    ax.set_title(f"{joint}: delay={rec['delay_frames']}f ({delay_ms:.0f}ms)  "
                f"tau={rec['tau_frames']}f ({tau_ms:.0f}ms)  "
                f"settle={rec['settling_frames']}f ({settle_ms:.0f}ms)", fontsize=9)
    ax.set_xlabel("frame"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = outdir / f"step_response_{joint}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def plot_memory_summary(per_joint: dict, rec: dict, cfg: dict, outdir: pathlib.Path) -> pathlib.Path | None:
    joints = [j for j, e in per_joint.items() if e["delay_frames"]]
    if not joints:
        return None
    delays = [per_joint[j]["delay_frames"]["median"] for j in joints]
    settles = [per_joint[j]["settling_frames"]["median"] if per_joint[j]["settling_frames"] else 0
               for j in joints]
    worst = rec.get("worst_case_joint")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(joints))
    delay_bars = ax.bar(x, delays, color=_BLUE, label="delay (median)")
    settle_bars = ax.bar(x, settles, bottom=delays, color=_AQUA, label="settling (median)")
    if worst in joints:
        wi = joints.index(worst)
        for bars in (delay_bars, settle_bars):
            bars[wi].set_edgecolor(_RED)
            bars[wi].set_linewidth(2)
    ax.axhline(cfg["current_obs_frames"], color=_MUTED, ls="--", lw=1,
               label=f"current_obs_frames={cfg['current_obs_frames']}")
    if rec.get("suggested_floor_frames"):
        ax.axhline(rec["suggested_floor_frames"], color=_INK, ls=":", lw=1,
                   label=f"suggested floor={rec['suggested_floor_frames']}")
    ax.set_xticks(x); ax.set_xticklabels(joints, rotation=90, fontsize=7)
    ax.set_ylabel("frames")
    ax.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=_GRID, lw=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "memory_summary.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def _strip_for_json(per_joint: dict) -> dict:
    out = {}
    for j, e in per_joint.items():
        e = dict(e)
        e.pop("_gt_records", None)
        e.pop("_ctrl_records", None)
        e["ctrl_side"] = {k: v for k, v in e["ctrl_side"].items()
                          if k not in ("records",)}
        out[j] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=False,
                    help="aligned *.aligned.npz glob(s) and/or raw .bag path(s)")
    ap.add_argument("--out", default=None, help="output dir (default: outputs/)")
    ap.add_argument("--live", action="store_true",
                    help="issue step commands on live hardware first (requires ROS + a "
                         "reachable hand; not implemented in this environment)")
    args = ap.parse_args()

    if args.live:
        print("--live requested: this would issue step commands to a few joints on the "
              "live hand and record the response before falling through to the offline "
              "analysis below. No ROS/hardware routine is implemented in this repo yet — "
              "run scripts/record_episode.sh with deliberate step commands instead, then "
              "pass the resulting bag/aligned file via --data.", file=sys.stderr)
        return 1

    if not args.data:
        print("no --data given; pass *.aligned.npz or a raw .bag (see --help)",
              file=sys.stderr)
        return 1

    cfg = cl.load_latency()
    joints = cl.load_joints()
    outdir = pathlib.Path(args.out) if args.out else cl.REPO_ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.data)
    if not episodes:
        print("no data files resolved from --data", file=sys.stderr)
        return 1
    rate = check_rate(episodes, float(cfg["dataset_rate"]))

    per_joint = build_joint_reports(episodes, joints, cfg, rate)
    total_steps = sum(e["n_steps"] for e in per_joint.values())
    if total_steps == 0:
        msg = (
            "No clean step commands were found in this data (looked for |delta "
            f"set_point| > {cfg['step_thresh']} rad holding for >= {cfg['hold_frames']} "
            "frames, per config/latency.yaml). This diagnostic needs deliberate step "
            "commands: command a joint to hold, jump to a new target, hold ~1s, per "
            "joint. See COLLECTION_PROTOCOL.md and re-record with that motion pattern, "
            "then re-run this script."
        )
        print(msg)
        report = {"dataset_rate_observed": rate, "dataset_rate_expected": cfg["dataset_rate"],
                  "n_episodes": len(episodes), "per_joint": {}, "verdict": "NO STEPS FOUND",
                  "message": msg}
        (outdir / "latency_report.json").write_text(json.dumps(report, indent=2))
        return 2

    rec = recommend(per_joint, cfg, rate)
    window_limited_joints = sorted(
        j for j, e in per_joint.items()
        if e.get("n_window_limited") or e["ctrl_side"].get("n_window_limited"))

    report = {
        "dataset_rate_observed": rate, "dataset_rate_expected": cfg["dataset_rate"],
        "n_episodes": len(episodes),
        "per_joint": _strip_for_json(per_joint),
        **rec,
        "current_obs_frames": cfg["current_obs_frames"],
        "current_action_frames": cfg["current_action_frames"],
        "window_limited_joints": window_limited_joints,
    }
    if window_limited_joints:
        report["notes"] = [
            f"response_window={cfg['response_window']} frames closed before the signal "
            f"fully stabilized for: {', '.join(window_limited_joints)}. Their settling/tau "
            "(and therefore the suggested floor) may be UNDERESTIMATED — increase "
            "response_window in config/latency.yaml and re-run to confirm."
        ]
    (outdir / "latency_report.json").write_text(json.dumps(report, indent=2, default=float))

    n_pngs = 0
    if _HAVE_MPL:
        for j, e in per_joint.items():
            if plot_step_response(j, e, episodes, joints, rate, outdir):
                n_pngs += 1
        plot_memory_summary(per_joint, rec, cfg, outdir)
    else:
        print("matplotlib unavailable — skipping PNGs, JSON report still written",
              file=sys.stderr)

    n_joints_with_data = sum(1 for e in per_joint.values() if e["n_ok"] > 0)
    print(f"\n{total_steps} step(s) detected across {len(episodes)} episode(s), "
          f"{n_joints_with_data}/{len(per_joint)} joints have >=1 usable step.")
    print(f"Data rate observed: {rate:g} Hz (config expects {cfg['dataset_rate']:g} Hz).")
    if window_limited_joints:
        print(f"WARNING: response_window may be too short for: "
              f"{', '.join(window_limited_joints)} (settling/tau likely underestimated "
              f"— see 'notes' in the JSON report).")
    print(f"Verdict: {rec['verdict']}")
    print(f"wrote {outdir / 'latency_report.json'}"
          + (f", {n_pngs} step_response PNG(s), memory_summary.png" if _HAVE_MPL else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
