#!/usr/bin/env python3
"""Hardware-side twin of collect_sim2real_data.py.

Loads pre-recorded sim rollout NPZ files (from results/tactile_characterization/)
and replays the exact same sine-wave joint commands on the Shadow Lite hand via ROS,
while logging the full 64-taxel TouchLab tactile sensor with configurable clustering.

Output NPZ schema matches the sim side so analyze_tactile.py works without changes.
For direct sim-vs-hardware comparison via analyze_tactile.py, use --clusters_per_finger 1
(produces tactile shape (T, 4), one value per finger, same as sim).

Usage:
    python run_simgap_hardware.py --data_dir results/tactile_characterization/
    python run_simgap_hardware.py --data_dir results/tactile_characterization/ --seeds 0-4
    python run_simgap_hardware.py --data_dir results/tactile_characterization/ --clusters_per_finger 1 --agg mean
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False

# =============================================================================
# JOINT LAYOUT — copied verbatim from run_shadow.py
# =============================================================================

POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # policy index 0
    "rh_MFJ4",   # policy index 1
    "rh_RFJ4",   # policy index 2
    "rh_THJ5",   # policy index 3
    "rh_FFJ3",   # policy index 4
    "rh_MFJ3",   # policy index 5
    "rh_RFJ3",   # policy index 6
    "rh_THJ4",   # policy index 7
    "rh_FFJ2",   # policy index 8  — represents ffj0 (J2 is the controller in sim)
    "rh_MFJ2",   # policy index 9  — represents mfj0
    "rh_RFJ2",   # policy index 10 — represents rfj0
    "rh_THJ2",   # policy index 11
    "rh_THJ1",   # policy index 12
]

# Maps subscriber (JointState) index → policy index
INDEX_RESHUFFLE_MAP = {
    3:  0,   # FFJ4
    7:  1,   # MFJ4
    11: 2,   # RFJ4
    15: 3,   # THJ5
    2:  4,   # FFJ3
    6:  5,   # MFJ3
    10: 6,   # RFJ3
    14: 7,   # THJ4
    1:  8,   # FFJ2 (J2 = controller)
    5:  9,   # MFJ2
    9:  10,  # RFJ2
    13: 11,  # THJ2
    12: 12,  # THJ1
}

LOWER_LIMITS = np.array([
    -0.3491, -0.3491, -0.3491, -1.0472,
    -0.2618, -0.2618, -0.2618,  0.0,
     0.0,     0.0,     0.0,    -0.6981, -0.2618,
], dtype=np.float32)

UPPER_LIMITS = np.array([
    0.3491, 0.3491, 0.3491, 1.0472,
    1.5708, 1.5708, 1.5708, 1.2217,
    1.5708, 1.5708, 1.5708, 0.6981, 1.5708,
], dtype=np.float32)

VEL_LIMITS_NORM = np.array([
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0, 2.0, 4.0,
], dtype=np.float32)

# Default (ball-holding) pose from roto/tasks/robots/shadowlite/shadowlite.py
# Order matches POLICY_JOINT_ORDER: FFJ4, MFJ4, RFJ4, THJ5, FFJ3, MFJ3, RFJ3, THJ4,
#                                   FFJ2, MFJ2, RFJ2, THJ2, THJ1
DEFAULT_JOINT_POS = np.array([
    -0.349,  # rh_FFJ4
     0.0,    # rh_MFJ4  (not set in cfg → 0)
    -0.349,  # rh_RFJ4
     0.4,    # rh_THJ5
     0.65,   # rh_FFJ3
     0.65,   # rh_MFJ3
     0.65,   # rh_RFJ3
     0.5,    # rh_THJ4
     0.87,   # rh_FFJ2
     0.87,   # rh_MFJ2
     0.87,   # rh_RFJ2
     0.35,   # rh_THJ2
     0.0,    # rh_THJ1  (not set in cfg → 0)
], dtype=np.float32)

# =============================================================================
# TACTILE SENSOR LAYOUT
# =============================================================================
# The hardware streams 5 fingers, each 16 taxels, in this firmware order:
#     ff(0-15), mf(16-31), rf(32-47), lf(48-63), th(64-79)   -> 80 taxels
# LF (little finger) does not exist on the Shadow Lite and reads all zeros;
# we drop it and keep the 4 real fingers in sim order (ff, mf, rf, th).
# NOTE: thumb is the 5th segment (64-79), NOT 48-63 -- that block is LF zeros.
#
# The calibrated topic may arrive either:
#   - flat: one value per taxel       -> 80 values
#   - triplet-packed [0, 0, v, ...]   -> 240 values (real value = 3rd of each triplet)
# Either way we reduce to 80 taxels in hardware order before slicing.

TAXELS_PER_FINGER = 16
HW_FINGER_ORDER   = ["ff", "mf", "rf", "lf", "th"]              # firmware stream order (LF = zeros on Lite)
N_HW_TAXELS       = len(HW_FINGER_ORDER) * TAXELS_PER_FINGER    # 80

_FINGER_SLICES = {
    name: slice(i * TAXELS_PER_FINGER, (i + 1) * TAXELS_PER_FINGER)
    for i, name in enumerate(HW_FINGER_ORDER)
}
# Sim output keeps the 4 real fingers, LF dropped:
_HW_TO_SIM_FINGER_ORDER = ["ff", "mf", "rf", "th"]

TACTILE_TOPIC = "/shadow_touchlab_translator/calibrated_flat"


# =============================================================================
# TAXEL CLUSTERER
# =============================================================================

class TaxelClusterer:
    """Clusters 16 taxels per finger into C groups and reorders to sim channel order.

    Input:  (80,) taxels in hardware order: ff(0-15), mf(16-31), rf(32-47), lf(48-63), th(64-79)
    Output: (4*C,) clustered values in sim order: ff, mf, rf, th — each with C values (LF dropped)

    Grouping: C consecutive equal-sized groups per finger.
    E.g. C=4 → groups [0-3], [4-7], [8-11], [12-15] per finger.
    Remainder taxels are appended to the last group.

    For direct compatibility with analyze_tactile.py (which expects 4 channels),
    use clusters_per_finger=1.
    """

    def __init__(self, clusters_per_finger: int = 4, agg: str = "mean"):
        assert clusters_per_finger >= 1
        assert agg in ("mean", "sum", "max"), f"agg must be mean/sum/max, got {agg}"
        self.C = clusters_per_finger
        self.agg = agg

        # Build group index lists for 16 taxels split into C groups
        n_taxels = 16
        group_size = n_taxels // clusters_per_finger
        self._groups: list[list[int]] = []
        for g in range(clusters_per_finger):
            start = g * group_size
            end = start + group_size if g < clusters_per_finger - 1 else n_taxels
            self._groups.append(list(range(start, end)))

    def _agg_fn(self, vals: np.ndarray) -> float:
        if self.agg == "mean":
            return float(vals.mean())
        if self.agg == "sum":
            return float(vals.sum())
        return float(vals.max())  # max

    def cluster(self, taxels_64: np.ndarray) -> np.ndarray:
        """taxels_64: (64,) in hardware order. Returns (4*C,) in sim order (ff,mf,rf,th)."""
        out = []
        for finger_name in _HW_TO_SIM_FINGER_ORDER:
            sl = _FINGER_SLICES[finger_name]
            finger_taxels = taxels_64[sl]   # (16,)
            for group_indices in self._groups:
                out.append(self._agg_fn(finger_taxels[group_indices]))
        return np.array(out, dtype=np.float32)


# =============================================================================
# MATH UTILITIES — copied from run_shadow.py
# =============================================================================

def normalise(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Maps [-1, 1] policy space → radians."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def reshuffle(data_list: list, mapping: dict) -> np.ndarray:
    """Reorder from JointState subscriber order to policy order (13 values)."""
    out = np.zeros(len(mapping), dtype=np.float32)
    for sub_idx, pol_idx in mapping.items():
        if sub_idx < len(data_list):
            out[pol_idx] = data_list[sub_idx]
    return out


# =============================================================================
# ROS PUBLISHERS
# =============================================================================

def create_hand_publishers() -> dict:
    controller_names = [
        "ffj0", "ffj3", "ffj4",
        "mfj0", "mfj3", "mfj4",
        "rfj0", "rfj3", "rfj4",
        "thj1", "thj2", "thj4", "thj5",
    ]
    pubs = {}
    for name in controller_names:
        topic = f"/sh_rh_{name}_position_controller/command"
        pubs[name] = rospy.Publisher(topic, Float64, queue_size=1)
        rospy.loginfo(f"Publisher: {topic}")
    return pubs


def publish_to_hand(pubs: dict, actions_radians: np.ndarray) -> None:
    """Send 13 joint commands (in radians, policy order) to hardware.

    Coupled joints (ffj0/mfj0/rfj0) are published with 2× multiplier:
    policy outputs 0–π/2 per coupled joint, hardware controller expects 0–π.
    """
    commands = {
        "ffj4": float(actions_radians[0]),
        "mfj4": float(actions_radians[1]),
        "rfj4": float(actions_radians[2]),
        "thj5": float(actions_radians[3]),
        "ffj3": float(actions_radians[4]),
        "mfj3": float(actions_radians[5]),
        "rfj3": float(actions_radians[6]),
        "thj4": float(actions_radians[7]),
        "thj2": float(actions_radians[11]),
        "thj1": float(actions_radians[12]),
        "ffj0": 2.0 * float(actions_radians[8]),
        "mfj0": 2.0 * float(actions_radians[9]),
        "rfj0": 2.0 * float(actions_radians[10]),
    }
    for name, val in commands.items():
        msg = Float64()
        msg.data = val
        pubs[name].publish(msg)


def publish_default_pose(pubs: dict, duration_s: float = 3.0) -> None:
    """Move all joints to the Baoding ball-holding default pose and hold for duration_s."""
    rospy.loginfo(f"Moving to default (ball-holding) pose for {duration_s}s ...")
    rate = rospy.Rate(10)
    n = int(duration_s * 10)
    for _ in range(n):
        publish_to_hand(pubs, DEFAULT_JOINT_POS)
        rate.sleep()
    rospy.loginfo("Default pose reached.")


def wait_for_ball_placement(duration_s: float = 10.0) -> None:
    """Hold default pose and count down so the user can place the ball."""
    rospy.loginfo(
        f"\n{'='*60}\n"
        f"  PLACE THE BALL NOW  —  holding default pose for {duration_s:.0f}s\n"
        f"{'='*60}"
    )
    for remaining in range(int(duration_s), 0, -1):
        if rospy.is_shutdown():
            break
        rospy.loginfo(f"  Starting in {remaining}s ...")
        rospy.sleep(1.0)
    rospy.loginfo("Starting replay!")


# =============================================================================
# SENSOR STATE (shared between callbacks and main loop)
# =============================================================================

_lock = Lock()
_joint_pos: np.ndarray | None = None   # (13,) radians, policy order
_joint_vel: np.ndarray | None = None   # (13,) rad/s, policy order
_tactile_hw: np.ndarray | None = None  # (80,) taxel values, hardware order (ff,mf,rf,lf,th)


def _prop_callback(msg: JointState) -> None:
    global _joint_pos, _joint_vel
    pos = reshuffle(list(msg.position), INDEX_RESHUFFLE_MAP)
    vel = reshuffle(list(msg.velocity), INDEX_RESHUFFLE_MAP)
    with _lock:
        _joint_pos = pos
        _joint_vel = vel


def _tactile_callback(msg) -> None:
    global _tactile_hw
    # Topic may be a stamped wrapper (msg.multi_array.data) or plain (msg.data).
    raw_data = msg.multi_array.data if hasattr(msg, "multi_array") else msg.data
    raw = np.array(list(raw_data), dtype=np.float32)
    n = len(raw)
    if n == 3 * N_HW_TAXELS:        # 240 — triplet-packed [0, 0, v, ...]
        taxels = raw[2::3]          # 3rd element of each triplet → (80,)
    elif n == N_HW_TAXELS:          # 80 — one value per taxel
        taxels = raw
    else:
        rospy.logwarn_throttle(
            5.0, f"Tactile message has {n} values, expected {N_HW_TAXELS} or {3 * N_HW_TAXELS}")
        return
    with _lock:
        _tactile_hw = taxels


def _wait_for_sensors(timeout_s: float = 10.0, no_tactile: bool = False) -> None:
    """Block until at least one message has arrived on required topics."""
    msg = "joint_states" if no_tactile else "joint_states and tactile"
    rospy.loginfo(f"Waiting for {msg} ...")
    deadline = time.time() + timeout_s
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        with _lock:
            got_joints = _joint_pos is not None
            got_tactile = _tactile_hw is not None
        ready = got_joints if no_tactile else (got_joints and got_tactile)
        if ready:
            rospy.loginfo("Sensors ready.")
            return
        if time.time() > deadline:
            missing = []
            if not got_joints:
                missing.append("/joint_states")
            if not no_tactile and not got_tactile:
                missing.append(TACTILE_TOPIC)
            rospy.logerr(f"Timeout waiting for: {', '.join(missing)}")
            if not no_tactile:
                rospy.logerr(
                    "Checklist:\n"
                    "  (1) Translator node running: "
                    "roslaunch touchlab_driver_ros translator.launch\n"
                    "       (source /opt/ros/noetic/setup.bash && source /ros1/devel/setup.bash first)\n"
                    "  (2) Relay bridge running: python /ros1/src/tactile_relay.py\n"
                    "       (same /ros1 environment — needed to convert stamped → flat msg)\n"
                    "  (3) Verify with: rostopic hz /shadow_touchlab_translator/calibrated_flat"
                )
            raise RuntimeError(f"Timeout waiting for: {', '.join(missing)}")
        rate.sleep()


# =============================================================================
# SEED PARSING
# =============================================================================

def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse "0-29", "0,5,10", or "3" into a list of ints."""
    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


# =============================================================================
# NPZ SAVING
# =============================================================================

def _save_npz(
    out_dir: Path,
    tag: str,
    seed: int,
    actions: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    tactile_clustered: np.ndarray,
    tactile_raw: np.ndarray,
    timestamps: np.ndarray,
    dt: float,
    clusterer: TaxelClusterer,
    sim_npz: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_noball_seed{seed:04d}.npz"
    payload = dict(
        actions=actions,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        tactile=tactile_clustered,       # (T, 4*C) in ff/mf/rf/th order — analyze_tactile.py key
        tactile_raw=tactile_raw,         # (T, 80) hardware order (ff/mf/rf/lf/th); LF segment is zeros
        timestamps=timestamps,
        joint_names=np.array(POLICY_JOINT_ORDER),
        seed=seed,
        dt=dt,
        duration_s=float(len(actions) * dt),
        with_ball=False,
        tag=tag,
        clusters_per_finger=clusterer.C,
        agg_method=clusterer.agg,
    )
    # Copy trajectory metadata from sim NPZ for alignment verification
    for key in ("traj_amps", "traj_freqs", "traj_phases"):
        if key in sim_npz:
            payload[key] = sim_npz[key]
    np.savez_compressed(out, **payload)
    rospy.loginfo(f"Saved {out}  (T={len(actions)}, tactile={tactile_clustered.shape})")
    return out


# =============================================================================
# LIVE TACTILE PLOT
# =============================================================================

class LiveTactilePlot:
    _FINGER_LABELS = ["FF", "MF", "RF", "TH"]
    _FINGER_COLORS = ["steelblue", "seagreen", "tomato", "darkorange"]

    def __init__(self, clusters_per_finger: int):
        self.C = clusters_per_finger
        n = 4 * clusters_per_finger
        colors, labels = [], []
        for name, col in zip(self._FINGER_LABELS, self._FINGER_COLORS):
            for c in range(clusters_per_finger):
                colors.append(col)
                labels.append(f"{name}_{c}" if clusters_per_finger > 1 else name)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n)
        self.bars = self.ax.bar(x, np.zeros(n), color=colors)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, fontsize=9)
        self.ax.set_ylabel("Tactile value")
        self.ax.set_ylim(0, 1.0)
        self.fig.tight_layout()
        plt.pause(0.05)

    def update(self, clustered: np.ndarray, step: int, T: int, t: float, seed: int) -> None:
        for bar, val in zip(self.bars, clustered):
            bar.set_height(float(val))
        peak = float(clustered.max())
        if peak > 0:
            self.ax.set_ylim(0, peak * 1.3)
        self.fig.suptitle(
            f"Tactile contact  |  seed={seed}  step={step}/{T}  t={t:.1f}s",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self.fig)


# =============================================================================
# MAIN
# =============================================================================

def _load_npz_single(npz_path: Path):
    """Load a single-file NPZ (from record_policy.py or sim_noball format).

    Returns (actions_norm [T,13], dt, sim_data dict).
    Supports both 'dt' (sim_noball) and 'rl_dt' (record_policy) keys.
    """
    sim_data = np.load(npz_path, allow_pickle=False)
    actions_norm = sim_data["actions"].astype(np.float32)   # (T, 13) in [-1,1]
    if "dt" in sim_data:
        dt = float(sim_data["dt"])
    elif "rl_dt" in sim_data:
        dt = float(sim_data["rl_dt"])
    else:
        raise KeyError(f"NPZ {npz_path} has neither 'dt' nor 'rl_dt' key")
    return actions_norm, dt, sim_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay sim trajectories on hardware and log joint data."
    )
    # --- input mode: single NPZ or directory of seeds ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", type=Path, default=None,
                       help="Single NPZ file to replay (from record_policy.py or sim_noball format). "
                            "Bypasses --data_dir / --seeds loop; uses --tag for output name.")
    group.add_argument("--data_dir", type=Path, default=None,
                       help="Directory containing sim_noball_seed*.npz files.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: results/hardware_characterization/).")
    parser.add_argument("--seeds", default="0-29",
                        help="Seeds to run (data_dir mode): '0-29', '0,5,10', or '3'.")
    parser.add_argument("--clusters_per_finger", type=int, default=1,
                        help="Taxels per finger to cluster into (default: 1). "
                             "Use 1 for direct analyze_tactile.py compatibility.")
    parser.add_argument("--agg", default="sum", choices=["mean", "sum", "max"],
                        help="Taxel aggregation method within each cluster (default: sum).")
    parser.add_argument("--tag", default="real", help="Output filename prefix (default: real).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed fraction (default: 1.0, e.g. 0.3 = 30%% speed).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the live tactile contact plot.")
    parser.add_argument("--no_tactile", action="store_true",
                        help="Skip tactile sensor subscription and recording. "
                             "Use when TouchLab translator is not running (joints-only prop-gap run).")
    parser.add_argument("--ball_wait", type=float, default=10.0,
                        help="Seconds to hold default pose before each replay so you can place "
                             "the ball (default: 10).")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("results/hardware_characterization")
    clusterer = TaxelClusterer(args.clusters_per_finger, args.agg)
    no_tactile: bool = args.no_tactile

    rospy.init_node("simgap_hardware_collector", anonymous=False)

    # Publishers
    pubs = create_hand_publishers()
    rospy.sleep(0.5)   # let ROS connect publishers

    # Subscribers
    rospy.Subscriber("/joint_states", JointState, _prop_callback, queue_size=1)
    if not no_tactile:
        rospy.Subscriber(TACTILE_TOPIC, Float64MultiArray, _tactile_callback, queue_size=1)

    _wait_for_sensors(no_tactile=no_tactile)
    publish_default_pose(pubs, duration_s=3.0)

    plot: LiveTactilePlot | None = None
    if not args.no_plot and not no_tactile:
        if _MATPLOTLIB_OK:
            plot = LiveTactilePlot(clusterer.C)
        else:
            rospy.logwarn("matplotlib unavailable — live plot disabled. Check DISPLAY / TkAgg backend.")

    # --- build list of (seed, npz_path) pairs to iterate ---
    if args.npz is not None:
        rollout_list = [(0, args.npz)]   # single-file mode: use seed=0 for output naming
        rospy.loginfo(f"Single-file mode: {args.npz}")
    else:
        seeds = _parse_seeds(args.seeds)
        rollout_list = [(s, args.data_dir / f"sim_noball_seed{s:04d}.npz") for s in seeds]

    rospy.loginfo(
        f"Starting {len(rollout_list)} rollout(s) | "
        f"clusters_per_finger={clusterer.C} | agg={clusterer.agg} | "
        f"tag={args.tag} | speed={args.speed} | no_tactile={no_tactile}"
    )

    for rollout_idx, (seed, npz_path) in enumerate(rollout_list):
        if rospy.is_shutdown():
            break

        # --- Load sim NPZ ---
        if not npz_path.exists():
            rospy.logwarn(f"NPZ not found: {npz_path}, skipping")
            continue

        actions_norm, dt, sim_data = _load_npz_single(npz_path)
        T = actions_norm.shape[0]
        hz = round(1.0 / dt)

        # Scale to radians for hardware
        actions_rad = scale(actions_norm, LOWER_LIMITS, UPPER_LIMITS)   # (T, 13)
        # Clip to safe limits (radians)
        actions_rad = np.clip(actions_rad, LOWER_LIMITS, UPPER_LIMITS)

        # --- Manual confirmation ---
        rospy.loginfo(
            f"\n{'='*60}\n"
            f"  Rollout {rollout_idx+1}/{len(rollout_list)} — seed {seed}\n"
            f"  T={T} steps | dt={dt:.4f}s ({hz} Hz) | ~{T*dt:.1f}s\n"
            f"{'='*60}"
        )
        try:
            input("  Press Enter to start, Ctrl+C to abort: ")
        except KeyboardInterrupt:
            rospy.loginfo("Aborted by user.")
            break

        # Hold default pose and count down so user can place the ball
        publish_default_pose(pubs, duration_s=3.0)
        wait_for_ball_placement(duration_s=args.ball_wait)

        # --- Allocate buffers ---
        buf_actions        = np.zeros((T, 13), dtype=np.float32)
        buf_joint_pos      = np.zeros((T, 13), dtype=np.float32)
        buf_joint_vel      = np.zeros((T, 13), dtype=np.float32)
        buf_tactile_raw    = np.zeros((T, N_HW_TAXELS), dtype=np.float32)
        buf_tactile_clust  = np.zeros((T, 4 * clusterer.C), dtype=np.float32)
        buf_timestamps     = np.zeros(T, dtype=np.float64)

        # --- Replay loop ---
        rate = rospy.Rate(max(1.0, hz * args.speed))
        t0 = rospy.Time.now().to_sec()

        for step in range(T):
            if rospy.is_shutdown():
                break

            cmd_rad = actions_rad[step]
            publish_to_hand(pubs, cmd_rad)

            # Read sensors under lock
            with _lock:
                jpos = _joint_pos.copy() if _joint_pos is not None else np.zeros(13, dtype=np.float32)
                jvel = _joint_vel.copy() if _joint_vel is not None else np.zeros(13, dtype=np.float32)
                if not no_tactile:
                    tac = _tactile_hw.copy() if _tactile_hw is not None else np.zeros(N_HW_TAXELS, dtype=np.float32)
                else:
                    tac = np.zeros(N_HW_TAXELS, dtype=np.float32)

            buf_actions[step]       = actions_norm[step]   # store original [-1,1] to match sim NPZ
            buf_joint_pos[step]     = jpos
            buf_joint_vel[step]     = jvel
            buf_tactile_raw[step]   = tac
            buf_tactile_clust[step] = clusterer.cluster(tac)
            buf_timestamps[step]    = rospy.Time.now().to_sec() - t0

            if plot is not None:
                plot.update(buf_tactile_clust[step], step, T, buf_timestamps[step], seed)

            if step % 60 == 0:
                rospy.loginfo(f"  step {step}/{T} | t={buf_timestamps[step]:.2f}s")

            rate.sleep()

        # --- Save ---
        _save_npz(
            out_dir=out_dir,
            tag=args.tag,
            seed=seed,
            actions=buf_actions,
            joint_pos=buf_joint_pos,
            joint_vel=buf_joint_vel,
            tactile_clustered=buf_tactile_clust,
            tactile_raw=buf_tactile_raw,
            timestamps=buf_timestamps,
            dt=dt,
            clusterer=clusterer,
            sim_npz=sim_data,
        )

        publish_default_pose(pubs, duration_s=2.0)

    if plot is not None:
        plot.close()

    rospy.loginfo("All rollouts complete.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except KeyboardInterrupt:
        rospy.loginfo("Aborted.")
        sys.exit(0)
