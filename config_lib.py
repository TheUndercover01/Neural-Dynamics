"""Shared config + canonical-layout helpers for the actuator-net pipeline.

Every script (collection, preprocess, QC) imports this so there is exactly ONE definition
of the actuator order, the frame layout, and the topic-name derivation rules.

Usage from any subdir:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import config_lib as cl
"""
from __future__ import annotations

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"

# --- Frame / sample geometry (the single source of truth for the "208" and "52") -------
# Frame layout mirrors the RL policy's observation (get_proprioception):
#   [ pos_norm(13) | vel_norm(13) | err(13) | action(13) ]  in POLICY joint order.
# pos/vel/action are normalised to [-1,1] by the policy limits; err is raw radians.
FRAME_FEATURE_GROUPS = ("pos_norm", "vel_norm", "err", "action")  # order within a frame
N_ACTUATORS = 13
N_JOINTS = 16
FRAME_DIM = len(FRAME_FEATURE_GROUPS) * N_ACTUATORS  # 4 * 13 = 52


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_joints() -> dict:
    d = _load_yaml(CONFIG_DIR / "joints.yaml")
    assert len(d["joint_order"]) == N_JOINTS, (
        f"joints.yaml joint_order has {len(d['joint_order'])} entries, expected {N_JOINTS}")
    assert len(d["actuator_order"]) == N_ACTUATORS, (
        f"joints.yaml actuator_order has {len(d['actuator_order'])} entries, "
        f"expected {N_ACTUATORS}")
    assert len(d["policy_joint_order"]) == N_ACTUATORS, (
        f"joints.yaml policy_joint_order has {len(d['policy_joint_order'])} entries, "
        f"expected {N_ACTUATORS}")
    for k in ("lower", "upper", "vel"):
        assert len(d["limits"][k]) == N_ACTUATORS, f"limits.{k} must have {N_ACTUATORS} entries"
    return d


def normalise(x, lower, upper):
    """Map [lower, upper] -> [-1, 1] (identical to normalise() in the policy node)."""
    import numpy as np
    return (2.0 * np.asarray(x) - np.asarray(upper) - np.asarray(lower)) / (
        np.asarray(upper) - np.asarray(lower))


def policy_perm(joints: dict | None = None) -> "list[int]":
    """Permutation p (len 13) mapping actuator_order columns -> policy_joint_order.

    data_policy_order[:, k] == data_actuator_order[:, p[k]].
    Coupled policy joints (FFJ2/MFJ2/RFJ2) are sourced from their J0 actuator.
    """
    joints = joints or load_joints()
    acts = list(joints["actuator_order"])
    p2a = dict(joints.get("policy_to_actuator", {}))
    perm = []
    for pj in joints["policy_joint_order"]:
        act = p2a.get(pj, pj)  # coupled -> J0 actuator; others map name-for-name
        assert act in acts, f"policy joint {pj} -> actuator {act} not in actuator_order"
        perm.append(acts.index(act))
    return perm


def policy_limits(joints: dict | None = None):
    """(lower[13], upper[13], vel[13]) arrays in policy_joint_order."""
    import numpy as np
    joints = joints or load_joints()
    lim = joints["limits"]
    return (np.asarray(lim["lower"], float),
            np.asarray(lim["upper"], float),
            np.asarray(lim["vel"], float))


def load_topics() -> dict:
    return _load_yaml(CONFIG_DIR / "topics.yaml")


def load_pipeline() -> dict:
    return _load_yaml(CONFIG_DIR / "pipeline.yaml")


def controller_state_topic(actuator: str) -> str:
    """rh_FFJ0 -> /sh_rh_ffj0_position_controller/state (lowercase)."""
    return f"/sh_{actuator.lower()}_position_controller/state"


def controller_command_topic(actuator: str) -> str:
    """rh_FFJ0 -> /sh_rh_ffj0_position_controller/command."""
    return f"/sh_{actuator.lower()}_position_controller/command"


def actuator_state_topics(joints: dict | None = None) -> "list[str]":
    joints = joints or load_joints()
    return [controller_state_topic(a) for a in joints["actuator_order"]]


def coupled_actuators(joints: dict | None = None) -> dict:
    joints = joints or load_joints()
    return dict(joints.get("coupling", {}))


def frame_column_names(joints: dict | None = None) -> "list[str]":
    """The 52 human-readable column names for one frame, in policy-frame order."""
    joints = joints or load_joints()
    pj = joints["policy_joint_order"]
    cols = []
    for group in FRAME_FEATURE_GROUPS:
        cols.extend(f"{group}:{j}" for j in pj)
    assert len(cols) == FRAME_DIM
    return cols


def input_column_names(joints: dict | None = None, stack_len: int = 4) -> "list[str]":
    """The full 208 input column names, oldest frame first, frame t last."""
    frame = frame_column_names(joints)
    cols = []
    for k in range(stack_len - 1, -1, -1):          # t-3, t-2, t-1, t
        cols.extend(f"t-{k}|{c}" if k else f"t|{c}" for c in frame)
    return cols
