from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional, Sequence

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mujoco
import numpy as np

from simadaptor.deploy.source_to_osc_common import (
    DEFAULT_OSC_NULLSPACE_STIFFNESS,
    DEFAULT_OSC_WAYPOINT_RPY_DEG_MAX,
    DEFAULT_OSC_WAYPOINT_RPY_DEG_MIN,
    DEFAULT_OSC_WAYPOINT_XYZ_MAX,
    DEFAULT_OSC_WAYPOINT_XYZ_MIN,
    HOME_Q,
    _history_controller_cartesian_error,
    _history_controller_damped_pseudoinverse,
    compute_target_metrics,
    make_cartesian_target_reference,
    make_source_reference,
    parse_vec3,
    parse_vec6,
    sample_osc_waypoints,
    fk_site_pose,
)


DEADZONE_SLOPE = 1e-2


DEFAULT_PROFILE_TABLE = Path("assets/datagen_profiles.json")
DEFAULT_PROFILE_KEY = "panda_pandagripper"
DEFAULT_PANDA_XML = Path("assets/franka_panda/panda_pandagripper.xml")
DEFAULT_PIPER_XML = Path("assets/piper/piper_description.xml")
DEFAULT_RBY1_ONEARM_XML = Path("assets/rby1a/rby1_onearm.xml")
DEFAULT_KUKA_IIWA14_XML = Path("assets/kuka_iiwa_14/iiwa14.xml")
DEFAULT_GOOGLE_ROBOT_XML = Path("assets/google_robot/google_robot.xml")
DEFAULT_UNITREE_Z1_XML = Path("assets/unitree_z1/unitree_z1.xml")
DEFAULT_FLEXIV_RIZON4_XML = Path("assets/flexiv_rizon4/flexiv_rizon4.xml")
DEFAULT_PIPER_HOME_Q = (0.0, 1.05, -1.25, 0.0, 0.55, 0.0)
DEFAULT_PIPER_SOURCE_AMP_DEG = (24.0, 22.0, 24.0, 22.0, 18.0, 22.0)
DEFAULT_PIPER_SOURCE_CYCLES = (3, 3, 3, 4, 4, 5)
DEFAULT_PIPER_JOINT_STIFFNESS = (35.0, 35.0, 30.0, 15.0, 15.0, 10.0)
DEFAULT_PIPER_JOINT_DAMPING = (6.0, 6.0, 5.0, 3.0, 3.0, 2.0)
DEFAULT_RBY1_ONEARM_HOME_Q = (0.0, 0.7, 0.0, -1.35, 0.0, 0.9, 0.0)
DEFAULT_RBY1_ONEARM_SOURCE_AMP_DEG = (22.0, 22.0, 24.0, 20.0, 24.0, 20.0, 28.0)
DEFAULT_RBY1_ONEARM_SOURCE_CYCLES = (3, 3, 3, 3, 4, 4, 5)
DEFAULT_RBY1_ONEARM_JOINT_STIFFNESS = (45.0, 45.0, 40.0, 35.0, 25.0, 20.0, 12.0)
DEFAULT_RBY1_ONEARM_JOINT_DAMPING = (8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)
DEFAULT_KUKA_IIWA14_HOME_Q = (0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0)
DEFAULT_KUKA_IIWA14_SOURCE_AMP_DEG = (24.0, 22.0, 24.0, 20.0, 22.0, 20.0, 26.0)
DEFAULT_KUKA_IIWA14_SOURCE_CYCLES = (3, 3, 3, 3, 4, 4, 5)
DEFAULT_KUKA_IIWA14_JOINT_STIFFNESS = (60.0, 60.0, 50.0, 40.0, 30.0, 20.0, 12.0)
DEFAULT_KUKA_IIWA14_JOINT_DAMPING = (10.0, 10.0, 9.0, 7.0, 5.0, 4.0, 3.0)
DEFAULT_GOOGLE_ROBOT_HOME_Q = (0.0, 0.6, 0.0, 1.2, 0.0, 0.5, 0.0)
DEFAULT_GOOGLE_ROBOT_SOURCE_AMP_DEG = (20.0, 22.0, 22.0, 20.0, 24.0, 20.0, 26.0)
DEFAULT_GOOGLE_ROBOT_SOURCE_CYCLES = (3, 3, 3, 3, 4, 4, 5)
DEFAULT_GOOGLE_ROBOT_JOINT_STIFFNESS = (50.0, 50.0, 40.0, 30.0, 20.0, 15.0, 10.0)
DEFAULT_GOOGLE_ROBOT_JOINT_DAMPING = (8.0, 8.0, 7.0, 6.0, 4.0, 3.0, 2.0)
DEFAULT_UNITREE_Z1_HOME_Q = (0.0, 0.785, -0.261, -0.523, 0.0, 0.0)
DEFAULT_UNITREE_Z1_SOURCE_AMP_DEG = (22.0, 22.0, 22.0, 20.0, 18.0, 24.0)
DEFAULT_UNITREE_Z1_SOURCE_CYCLES = (3, 3, 3, 4, 4, 5)
DEFAULT_UNITREE_Z1_JOINT_STIFFNESS = (25.0, 25.0, 20.0, 14.0, 10.0, 8.0)
DEFAULT_UNITREE_Z1_JOINT_DAMPING = (5.0, 5.0, 4.0, 3.0, 2.0, 2.0)
DEFAULT_FLEXIV_RIZON4_HOME_Q = (0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0)
DEFAULT_FLEXIV_RIZON4_SOURCE_AMP_DEG = (24.0, 22.0, 24.0, 20.0, 22.0, 20.0, 26.0)
DEFAULT_FLEXIV_RIZON4_SOURCE_CYCLES = (3, 3, 3, 3, 4, 4, 5)
DEFAULT_FLEXIV_RIZON4_JOINT_STIFFNESS = (55.0, 55.0, 45.0, 35.0, 25.0, 20.0, 12.0)
DEFAULT_FLEXIV_RIZON4_JOINT_DAMPING = (9.0, 9.0, 8.0, 6.0, 5.0, 4.0, 3.0)
DEFAULT_SIM_OSC_WAYPOINT_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.12, 0.08, 0.04),
    (0.24, -0.11, 0.11),
    (0.34, 0.13, 0.07),
)
DEFAULT_SIM_OSC_WAYPOINT_RPY_DEG: tuple[tuple[float, float, float], ...] = (
    (25.0, -20.0, 30.0),
    (-25.0, 28.0, -35.0),
    (35.0, -30.0, 55.0),
)


@dataclasses.dataclass(frozen=True)
class SimConditionSpec:
    key: str
    label: str
    source_segment: str
    switch_behavior: str
    target_segment: str
    tam_on_source: bool = False
    tam_on_target: bool = False
    reset_tam_at_switch: bool = False


TAM_CONDITIONS: tuple[SimConditionSpec, ...] = (
    SimConditionSpec(
        key="direct_osc",
        label="Direct OSC",
        source_segment="Same source warm-up",
        switch_behavior="TAM disabled",
        target_segment="OSC without TAM",
        tam_on_source=False,
        tam_on_target=False,
        reset_tam_at_switch=False,
    ),
    SimConditionSpec(
        key="tam_reset",
        label="TAM reset",
        source_segment="Joint impedance + TAM",
        switch_behavior="Clear history, cache, latent, short window",
        target_segment="OSC + cold-start TAM",
        tam_on_source=True,
        tam_on_target=True,
        reset_tam_at_switch=True,
    ),
    SimConditionSpec(
        key="tam_carried",
        label="TAM carried",
        source_segment="Joint impedance + TAM",
        switch_behavior="Keep TAM state",
        target_segment="OSC + carried TAM",
        tam_on_source=True,
        tam_on_target=True,
        reset_tam_at_switch=False,
    ),
)
IDEAL_SPEC = SimConditionSpec(
    key="ideal_model",
    label="Ideal model",
    source_segment="Same source warm-up",
    switch_behavior="Ideal physics, TAM disabled",
    target_segment="OSC ideal-model trajectory",
    tam_on_source=False,
    tam_on_target=False,
    reset_tam_at_switch=False,
)
SIM_CONDITIONS: tuple[SimConditionSpec, ...] = (
    *TAM_CONDITIONS,
)
SIM_CONDITION_BY_KEY = {spec.key: spec for spec in SIM_CONDITIONS}


def resolve_sim_conditions(raw_conditions: Sequence[str]) -> list[SimConditionSpec]:
    if not raw_conditions:
        return list(SIM_CONDITIONS)
    out: list[SimConditionSpec] = []
    aliases = {
        "all": "__all__",
        "table": "__all__",
        "tam_all": "__tam__",
        "tam": "__tam__",
        "direct": "direct_osc",
        "none": "direct_osc",
        "osc": "direct_osc",
        "tam_reset": "tam_reset",
        "tam_cold": "tam_reset",
        "reset": "tam_reset",
        "cold": "tam_reset",
        "tam_carried": "tam_carried",
        "tam_carry": "tam_carried",
        "carried": "tam_carried",
        "carry": "tam_carried",
    }
    for raw in raw_conditions:
        key = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if key == "__all__":
            for spec in SIM_CONDITIONS:
                if spec not in out:
                    out.append(spec)
            continue
        if key == "__tam__":
            for spec in TAM_CONDITIONS:
                if spec not in out:
                    out.append(spec)
            continue
        if key not in SIM_CONDITION_BY_KEY:
            choices = ", ".join(
                [
                    "all",
                    "tam_all",
                    "tam_reset",
                    "tam_carried",
                    *(spec.key for spec in SIM_CONDITIONS),
                ]
            )
            raise SystemExit(f"Unknown condition {raw!r}. Choices: {choices}")
        spec = SIM_CONDITION_BY_KEY[key]
        if spec not in out:
            out.append(spec)
    return out


def _condition_uses_tam(spec: SimConditionSpec) -> bool:
    return bool(spec.tam_on_source or spec.tam_on_target)


@dataclasses.dataclass(frozen=True)
class SimReference:
    iteration: int
    source_seed: int
    sim_seed: int
    source_amp_deg: tuple[float, ...]
    source_cycles: tuple[int, ...]
    osc_delta_xyz: tuple[float, float, float]
    osc_delta_rpy_deg: tuple[float, float, float]
    osc_waypoint_xyz: tuple[tuple[float, float, float], ...]
    osc_waypoint_rpy_deg: tuple[tuple[float, float, float], ...]
    source_t: np.ndarray
    source_q: np.ndarray
    source_dq: np.ndarray
    target_t: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray
    osc_start_pos: np.ndarray
    osc_start_quat: np.ndarray


@dataclasses.dataclass
class OnlineTamSimState:
    runtime: Any
    adaptor_apply_jit: Any
    model_params: Any
    norm_stats: Any
    min_patches_before_send: int
    embedding_interval_s: float
    enable_after_first_embedding: bool
    # base_tam_fusion checkpoints stream three torque histories (applied, base,
    # TAM residual) through weight-sharing encoders and a linear fusion layer,
    # mirroring mapping_server._push_fused_history_window.
    history_torque_mode: str = "applied"
    base_runtime: Any = None
    tam_runtime: Any = None
    history_fusion_params: Any = None
    patches_since_reset: int = 0
    num_embeddings: int = 0
    num_sent: int = 0
    enabled: bool = False
    last_send_t: float = -float("inf")
    current_embedding: Any = None

    @property
    def delay_enable(self) -> bool:
        return bool(self.enable_after_first_embedding or self.min_patches_before_send > 0)

    @property
    def uses_fused_history(self) -> bool:
        return str(self.history_torque_mode) == "base_tam_fusion"

    def reset(self) -> None:
        self.runtime.reset()
        if self.base_runtime is not None:
            self.base_runtime.reset()
        if self.tam_runtime is not None:
            self.tam_runtime.reset()
        self.patches_since_reset = 0
        self.num_embeddings = 0
        self.num_sent = 0
        self.enabled = not self.delay_enable
        self.last_send_t = -float("inf")
        self.current_embedding = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def parse_float_vec(raw: str) -> tuple[float, ...]:
    vals = [float(part) for part in str(raw).split(",") if part.strip() != ""]
    if not vals:
        raise argparse.ArgumentTypeError("vector must contain at least one comma-separated float")
    return tuple(vals)


def parse_optional_float_vec(raw: str) -> tuple[float, ...] | None:
    token = str(raw).strip().lower()
    if token in {"auto", "none", "null"}:
        return None
    return parse_float_vec(raw)


def _robot_key_from_xml(xml_path: Path) -> str:
    parts = {part.lower() for part in xml_path.parts}
    stem = xml_path.stem.lower()
    if "rby1a" in parts or stem in {"rby1", "rby1_onearm"}:
        return "rby1_onearm" if "onearm" in stem else "rby1"
    if "kuka_iiwa_14" in parts or stem in {"iiwa14", "kuka_iiwa14"}:
        return "iiwa14"
    if "google_robot" in parts or stem == "google_robot":
        return "google_robot"
    if "unitree_z1" in parts or stem in {"unitree_z1", "z1"}:
        return "unitree_z1"
    if "flexiv_rizon4" in parts or stem in {"flexiv_rizon4", "rizon4"}:
        return "flexiv_rizon4"
    if "piper" in parts or "piper" in stem:
        return "piper"
    if "franka_panda" in parts or "panda" in stem or "franka" in stem:
        return "panda"
    return stem


def _resize_float_vector(values: Sequence[float], dof: int, *, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values; got {arr.tolist()}.")
    if arr.size == 1:
        arr = np.repeat(arr, int(dof))
    elif arr.size < int(dof):
        arr = np.pad(arr, (0, int(dof) - int(arr.size)), constant_values=float(arr[-1]))
    elif arr.size > int(dof):
        arr = arr[: int(dof)]
    return tuple(float(x) for x in arr)


def _resize_int_vector(values: Sequence[int], dof: int, *, name: str) -> list[int]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if arr.size == 1:
        arr = np.repeat(arr, int(dof))
    elif arr.size < int(dof):
        arr = np.pad(arr, (0, int(dof) - int(arr.size)), constant_values=int(arr[-1]))
    elif arr.size > int(dof):
        arr = arr[: int(dof)]
    arr = np.maximum(arr, 1)
    return [int(x) for x in arr]


def _default_damping_ratio(dof: int) -> np.ndarray:
    return np.asarray(
        _resize_float_vector(
            (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
            dof,
            name="joint_damping_ratio",
        ),
        dtype=np.float64,
    )


def _actuated_hinge_joint_ids(model: mujoco.MjModel) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    actuator_trnid = np.asarray(model.actuator_trnid, dtype=np.int32)
    for actuator_idx in range(int(model.nu)):
        joint_id = int(actuator_trnid[actuator_idx, 0])
        if joint_id < 0 or joint_id >= int(model.njnt) or joint_id in seen:
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        seen.add(joint_id)
        out.append(joint_id)
    return out


def _arm_indices(model: mujoco.MjModel, dof: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    target_dof = int(dof) if dof is not None else int(model.nu)
    if target_dof <= 0:
        target_dof = 7
    joint_ids = [
        i
        for i in range(model.njnt)
        if (model.joint(i).name or "").startswith("panda_joint")
    ]
    if len(joint_ids) < target_dof:
        joint_ids = _actuated_hinge_joint_ids(model)
    if len(joint_ids) < target_dof:
        joint_ids = [
            i
            for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
    if len(joint_ids) < target_dof:
        raise ValueError(f"Model has only {len(joint_ids)} hinge joints; need {target_dof}.")
    joint_ids = joint_ids[:target_dof]
    qpos_idx = np.asarray([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    qvel_idx = np.asarray([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
    return qpos_idx, qvel_idx, joint_ids


def _default_initial_q(
    *,
    model: mujoco.MjModel,
    qpos_idx: np.ndarray,
    joint_ids: Sequence[int],
    xml_path: Path,
) -> tuple[float, ...]:
    dof = int(len(qpos_idx))
    robot_key = _robot_key_from_xml(xml_path)
    if robot_key == "piper":
        return _resize_float_vector(DEFAULT_PIPER_HOME_Q, dof, name="piper default initial_q")
    if robot_key == "rby1_onearm":
        return _resize_float_vector(DEFAULT_RBY1_ONEARM_HOME_Q, dof, name="rby1 default initial_q")
    if robot_key == "iiwa14":
        return _resize_float_vector(DEFAULT_KUKA_IIWA14_HOME_Q, dof, name="iiwa14 default initial_q")
    if robot_key == "google_robot":
        return _resize_float_vector(DEFAULT_GOOGLE_ROBOT_HOME_Q, dof, name="google_robot default initial_q")
    if robot_key == "unitree_z1":
        return _resize_float_vector(DEFAULT_UNITREE_Z1_HOME_Q, dof, name="unitree_z1 default initial_q")
    if robot_key == "flexiv_rizon4":
        return _resize_float_vector(DEFAULT_FLEXIV_RIZON4_HOME_Q, dof, name="flexiv_rizon4 default initial_q")
    if robot_key == "panda":
        return _resize_float_vector(HOME_Q, dof, name="panda default initial_q")

    q0 = np.asarray(model.qpos0[qpos_idx], dtype=np.float64).reshape(-1)
    if q0.shape[0] != dof:
        q0 = np.zeros((dof,), dtype=np.float64)
    if joint_ids:
        ranges = np.asarray(model.jnt_range[list(joint_ids)], dtype=np.float64).reshape(-1, 2)
        limited = np.asarray(model.jnt_limited[list(joint_ids)], dtype=bool).reshape(-1)
        finite = limited & np.all(np.isfinite(ranges), axis=1) & (ranges[:, 1] > ranges[:, 0])
        mid = 0.5 * (ranges[:, 0] + ranges[:, 1])
        q0 = np.where(finite, np.clip(q0, ranges[:, 0], ranges[:, 1]), q0)
        at_edge = finite & (
            (np.abs(q0 - ranges[:, 0]) < 1e-6)
            | (np.abs(q0 - ranges[:, 1]) < 1e-6)
        )
        q0 = np.where(at_edge, mid, q0)
    return tuple(float(x) for x in q0)


def _apply_robot_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "robot_preset", "auto") or "auto").lower()
    if preset in {"auto", ""}:
        return
    if preset == "panda":
        args.xml = DEFAULT_PANDA_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = DEFAULT_PROFILE_KEY
        return
    if preset == "piper":
        args.xml = DEFAULT_PIPER_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "piper_description"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_PIPER_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_PIPER_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_PIPER_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_PIPER_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_PIPER_JOINT_DAMPING
        return
    if preset in {"rby1", "rby1_onearm", "rby-1"}:
        args.xml = DEFAULT_RBY1_ONEARM_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "rby1_onearm"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_RBY1_ONEARM_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_RBY1_ONEARM_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_RBY1_ONEARM_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_RBY1_ONEARM_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_RBY1_ONEARM_JOINT_DAMPING
        return
    if preset in {"kuka", "kuka_iiwa14", "iiwa14"}:
        args.xml = DEFAULT_KUKA_IIWA14_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "iiwa14"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_KUKA_IIWA14_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_KUKA_IIWA14_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_KUKA_IIWA14_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_KUKA_IIWA14_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_KUKA_IIWA14_JOINT_DAMPING
        return
    if preset in {"google", "google_robot"}:
        args.xml = DEFAULT_GOOGLE_ROBOT_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "google_robot"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_GOOGLE_ROBOT_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_GOOGLE_ROBOT_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_GOOGLE_ROBOT_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_GOOGLE_ROBOT_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_GOOGLE_ROBOT_JOINT_DAMPING
        return
    if preset in {"unitree", "unitree_z1", "z1"}:
        args.xml = DEFAULT_UNITREE_Z1_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "unitree_z1"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_UNITREE_Z1_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_UNITREE_Z1_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_UNITREE_Z1_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_UNITREE_Z1_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_UNITREE_Z1_JOINT_DAMPING
        return
    if preset in {"flexiv", "flexiv_rizon4", "rizon4"}:
        args.xml = DEFAULT_FLEXIV_RIZON4_XML
        if getattr(args, "profile_key", None) is None:
            args.profile_key = "flexiv_rizon4"
        if getattr(args, "initial_q", None) is None:
            args.initial_q = DEFAULT_FLEXIV_RIZON4_HOME_Q
        if getattr(args, "source_amp_deg", None) is None:
            args.source_amp_deg = DEFAULT_FLEXIV_RIZON4_SOURCE_AMP_DEG
        if getattr(args, "source_cycles", None) is None:
            args.source_cycles = list(DEFAULT_FLEXIV_RIZON4_SOURCE_CYCLES)
        if getattr(args, "joint_stiffness", None) is None:
            args.joint_stiffness = DEFAULT_FLEXIV_RIZON4_JOINT_STIFFNESS
        if getattr(args, "joint_damping", None) is None:
            args.joint_damping = DEFAULT_FLEXIV_RIZON4_JOINT_DAMPING
        return
    raise SystemExit(
        f"Unknown --robot-preset={preset!r}; expected auto, panda, piper, rby1, "
        "iiwa14, google_robot, unitree_z1, or flexiv_rizon4."
    )


def _resolve_robot_args(args: argparse.Namespace) -> int:
    _apply_robot_preset(args)
    args.xml = args.xml.expanduser()
    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    requested_dof = None
    if getattr(args, "initial_q", None) is not None:
        requested_dof = int(np.asarray(args.initial_q, dtype=np.float64).reshape(-1).shape[0])
    qpos_idx, _qvel_idx, joint_ids = _arm_indices(model, dof=requested_dof)
    dof = int(len(qpos_idx))
    if dof <= 0:
        raise ValueError(f"Could not infer arm DoF from XML: {xml_path}")

    robot_key = _robot_key_from_xml(xml_path)
    if getattr(args, "profile_key", None) == "":
        args.profile_key = None
    if getattr(args, "profile_key", None) is None:
        default_profile_by_robot = {
            "piper": "piper_description",
            "rby1_onearm": "rby1_onearm",
            "iiwa14": "iiwa14",
            "google_robot": "google_robot",
            "unitree_z1": "unitree_z1",
            "flexiv_rizon4": "flexiv_rizon4",
        }
        args.profile_key = default_profile_by_robot.get(robot_key, None)

    if getattr(args, "initial_q", None) is None:
        args.initial_q = _default_initial_q(
            model=model,
            qpos_idx=qpos_idx,
            joint_ids=joint_ids,
            xml_path=xml_path,
        )
    else:
        initial_q = np.asarray(args.initial_q, dtype=np.float64).reshape(-1)
        if int(initial_q.shape[0]) != dof:
            raise ValueError(
                f"--initial-q length {initial_q.shape[0]} does not match inferred arm DoF {dof}."
        )
        args.initial_q = tuple(float(x) for x in initial_q)

    if getattr(args, "source_amp_deg", None) is None:
        source_amp_defaults = {
            "piper": DEFAULT_PIPER_SOURCE_AMP_DEG,
            "rby1_onearm": DEFAULT_RBY1_ONEARM_SOURCE_AMP_DEG,
            "iiwa14": DEFAULT_KUKA_IIWA14_SOURCE_AMP_DEG,
            "google_robot": DEFAULT_GOOGLE_ROBOT_SOURCE_AMP_DEG,
            "unitree_z1": DEFAULT_UNITREE_Z1_SOURCE_AMP_DEG,
            "flexiv_rizon4": DEFAULT_FLEXIV_RIZON4_SOURCE_AMP_DEG,
        }
        args.source_amp_deg = source_amp_defaults.get(
            robot_key,
            (30.0, 25.0, 30.0, 20.0, 30.0, 25.0, 34.0),
        )
    if getattr(args, "source_cycles", None) is None:
        source_cycle_defaults = {
            "piper": DEFAULT_PIPER_SOURCE_CYCLES,
            "rby1_onearm": DEFAULT_RBY1_ONEARM_SOURCE_CYCLES,
            "iiwa14": DEFAULT_KUKA_IIWA14_SOURCE_CYCLES,
            "google_robot": DEFAULT_GOOGLE_ROBOT_SOURCE_CYCLES,
            "unitree_z1": DEFAULT_UNITREE_Z1_SOURCE_CYCLES,
            "flexiv_rizon4": DEFAULT_FLEXIV_RIZON4_SOURCE_CYCLES,
        }
        args.source_cycles = list(source_cycle_defaults.get(robot_key, (3, 3, 3, 3, 3, 4, 6)))
    if getattr(args, "joint_stiffness", None) is None:
        stiffness_defaults = {
            "piper": DEFAULT_PIPER_JOINT_STIFFNESS,
            "rby1_onearm": DEFAULT_RBY1_ONEARM_JOINT_STIFFNESS,
            "iiwa14": DEFAULT_KUKA_IIWA14_JOINT_STIFFNESS,
            "google_robot": DEFAULT_GOOGLE_ROBOT_JOINT_STIFFNESS,
            "unitree_z1": DEFAULT_UNITREE_Z1_JOINT_STIFFNESS,
            "flexiv_rizon4": DEFAULT_FLEXIV_RIZON4_JOINT_STIFFNESS,
        }
        args.joint_stiffness = stiffness_defaults.get(
            robot_key,
            (50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 10.0),
        )
    if getattr(args, "joint_damping", None) is None:
        damping_defaults = {
            "piper": DEFAULT_PIPER_JOINT_DAMPING,
            "rby1_onearm": DEFAULT_RBY1_ONEARM_JOINT_DAMPING,
            "iiwa14": DEFAULT_KUKA_IIWA14_JOINT_DAMPING,
            "google_robot": DEFAULT_GOOGLE_ROBOT_JOINT_DAMPING,
            "unitree_z1": DEFAULT_UNITREE_Z1_JOINT_DAMPING,
            "flexiv_rizon4": DEFAULT_FLEXIV_RIZON4_JOINT_DAMPING,
        }
        args.joint_damping = damping_defaults.get(
            robot_key,
            (10.0, 10.0, 10.0, 8.0, 8.0, 8.0, 3.0),
        )

    args.initial_q = _resize_float_vector(args.initial_q, dof, name="initial_q")
    args.source_amp_deg = _resize_float_vector(args.source_amp_deg, dof, name="source_amp_deg")
    args.source_cycles = _resize_int_vector(args.source_cycles, dof, name="source_cycles")
    args.joint_stiffness = _resize_float_vector(args.joint_stiffness, dof, name="joint_stiffness")
    if args.joint_damping is not None:
        args.joint_damping = _resize_float_vector(args.joint_damping, dof, name="joint_damping")
    args.arm_dof = dof
    args.xml = xml_path
    return dof


def _site_quat_wxyz(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quat = np.zeros((4,), dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(9))
    return quat


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).reshape(4).copy()
    out[1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_error_rotvec(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_des = np.asarray(desired, dtype=np.float64).reshape(4)
    q_cur = np.asarray(current, dtype=np.float64).reshape(4)
    q_des = q_des / max(float(np.linalg.norm(q_des)), 1e-12)
    q_cur = q_cur / max(float(np.linalg.norm(q_cur)), 1e-12)
    q_err = _quat_mul(q_des, _quat_conj(q_cur))
    if q_err[0] < 0.0:
        q_err *= -1.0
    vec = q_err[1:]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-10:
        return 2.0 * vec
    angle = 2.0 * np.arctan2(vec_norm, float(q_err[0]))
    return angle * vec / vec_norm


def _set_arm_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    q: np.ndarray,
    dq: Optional[np.ndarray] = None,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
    if dq is not None:
        data.qvel[qvel_idx] = np.asarray(dq, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(model, data)


def _forward_controller_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    q: np.ndarray,
    dq: Optional[np.ndarray] = None,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
    if dq is not None:
        data.qvel[qvel_idx] = np.asarray(dq, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(model, data)


def _gravity_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    _forward_controller_state(model, data, qpos_idx, qvel_idx, q, np.zeros_like(q))
    return np.asarray(data.qfrc_bias[qvel_idx] - data.qfrc_gravcomp[qvel_idx], dtype=np.float64)


def _controller_side_guard_torque(
    *,
    q: np.ndarray,
    dq: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    joint_range: Optional[np.ndarray],
    enabled: bool = True,
    velocity_threshold: float = 4.0,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    dq = np.asarray(dq, dtype=np.float64).reshape(q.shape)
    kp = np.asarray(kp, dtype=np.float64).reshape(q.shape)
    kd = np.asarray(kd, dtype=np.float64).reshape(q.shape)
    if not bool(enabled):
        return np.zeros_like(q)

    tau = np.zeros_like(q)
    if joint_range is not None:
        range_arr = np.asarray(joint_range, dtype=np.float64)
        if range_arr.ndim > 2 and range_arr.shape[0] == 1:
            range_arr = range_arr[0]
        if range_arr.ndim >= 2 and range_arr.shape[-1] >= 2:
            range_arr = range_arr.reshape((-1, range_arr.shape[-1]))[: q.shape[0], :2]
            q_min = range_arr[:, 0]
            q_max = range_arr[:, 1]
            err_bound = np.where(
                q < q_min,
                q_min - q,
                np.where(q > q_max, q_max - q, 0.0),
            )
            tau = tau + (kp * 20.0) * err_bound

    high_vel_mask = np.abs(dq) > float(velocity_threshold)
    tau = tau - high_vel_mask * np.sign(dq) * (kd * 10.0) * (
        np.abs(dq) - float(velocity_threshold)
    )
    return tau


def _joint_impedance_torque(
    *,
    controller_model: mujoco.MjModel,
    controller_data: mujoco.MjData,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    joint_range: Optional[np.ndarray] = None,
    controller_guard_enabled: bool = True,
    velocity_threshold: float = 4.0,
) -> np.ndarray:
    return (
        np.asarray(kp, dtype=np.float64) * (np.asarray(q_ref, dtype=np.float64) - q)
        + np.asarray(kd, dtype=np.float64) * (np.asarray(dq_ref, dtype=np.float64) - dq)
        + _controller_side_guard_torque(
            q=q,
            dq=dq,
            kp=kp,
            kd=kd,
            joint_range=joint_range,
            enabled=controller_guard_enabled,
            velocity_threshold=velocity_threshold,
        )
        + _gravity_torque(controller_model, controller_data, qpos_idx, qvel_idx, q)
    )


def _osc_torque(
    *,
    plant_model: mujoco.MjModel,
    plant_data: mujoco.MjData,
    qpos_idx: np.ndarray,
    qvel_idx: np.ndarray,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    q_nullspace: np.ndarray,
    nullspace_stiffness: float,
    joint_kp: Optional[np.ndarray] = None,
    joint_kd: Optional[np.ndarray] = None,
    joint_range: Optional[np.ndarray] = None,
    controller_guard_enabled: bool = True,
    velocity_threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(plant_data.qpos[qpos_idx], dtype=np.float64)
    dq = np.asarray(plant_data.qvel[qvel_idx], dtype=np.float64)
    site_pos = np.asarray(plant_data.site_xpos[site_id], dtype=np.float64).copy()
    site_quat = _site_quat_wxyz(plant_data, site_id)
    goal_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)

    jacp = np.zeros((3, plant_model.nv), dtype=np.float64)
    jacr = np.zeros((3, plant_model.nv), dtype=np.float64)
    mujoco.mj_jacSite(plant_model, plant_data, jacp, jacr, site_id)
    jac = np.concatenate([jacp[:, qvel_idx], jacr[:, qvel_idx]], axis=0)

    cur_rot = np.asarray(plant_data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    cart_error = _history_controller_cartesian_error(
        current_pos=site_pos,
        current_quat_wxyz=site_quat,
        current_rot=cur_rot,
        target_pos=np.asarray(target_pos, dtype=np.float64).reshape(3),
        target_quat_wxyz=goal_quat,
    )
    task_velocity = jac @ dq
    kp = np.asarray(stiffness, dtype=np.float64).reshape(6)
    kd = np.asarray(damping, dtype=np.float64).reshape(6)
    tau_task = jac.T @ (-kp * cart_error - kd * task_velocity)

    jac_t_pinv = _history_controller_damped_pseudoinverse(jac.T)
    nullspace_projector = np.eye(q.shape[0], dtype=np.float64) - jac.T @ jac_t_pinv
    nullspace_k = float(nullspace_stiffness)
    nullspace_d = 2.0 * np.sqrt(max(nullspace_k, 0.0))
    tau_nullspace = nullspace_projector @ (
        nullspace_k * (np.asarray(q_nullspace, dtype=np.float64).reshape(-1) - q)
        - nullspace_d * dq
    )
    bias = np.asarray(plant_data.qfrc_bias[qvel_idx], dtype=np.float64)
    tau = tau_task + tau_nullspace + bias
    if joint_kp is not None and joint_kd is not None:
        tau = tau + _controller_side_guard_torque(
            q=q,
            dq=dq,
            kp=joint_kp,
            kd=joint_kd,
            joint_range=joint_range,
            enabled=controller_guard_enabled,
            velocity_threshold=velocity_threshold,
        )
    return np.asarray(tau, dtype=np.float64), site_pos, site_quat


def _as_np_field(x: Any, *, dtype=np.float64) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _remove_mujoco_robot_limits(model: mujoco.MjModel) -> None:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    for field_name in (
        "dof_frictionloss",
        "jnt_limited",
        "tendon_limited",
        "actuator_ctrllimited",
        "actuator_actlimited",
        "actuator_forcelimited",
        "jnt_actfrclimited",
    ):
        if hasattr(model, field_name):
            getattr(model, field_name)[:] = 0

    for field_name in (
        "jnt_range",
        "tendon_range",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_actrange",
        "jnt_actfrcrange",
    ):
        if hasattr(model, field_name):
            value = getattr(model, field_name)
            if getattr(value, "ndim", 0) >= 2 and value.shape[-1] >= 2:
                value[..., 0] = -np.inf
                value[..., 1] = np.inf

    if hasattr(model, "actuator_gear") and model.actuator_gear.size:
        model.actuator_gear[:] = 0.0
        model.actuator_gear[:, 0] = 1.0


def _zero_mjx_body_gravcomp(model: Any) -> Any:
    body_gravcomp = getattr(model, "body_gravcomp", None)
    if body_gravcomp is None:
        return model
    import jax.numpy as jnp

    zeros = jnp.zeros_like(body_gravcomp)
    if hasattr(model, "replace"):
        return model.replace(body_gravcomp=zeros)
    if hasattr(model, "tree_replace"):
        return model.tree_replace({"body_gravcomp": zeros})
    return model


def _fit_prefix(arr: Optional[np.ndarray], target: np.ndarray, axis: int = -1) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=target.dtype)
    axis = axis if axis >= 0 else arr.ndim + axis
    n = min(arr.shape[axis], target.shape[axis])
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, n)
    return np.asarray(arr[tuple(slicer)], dtype=target.dtype)


def _apply_rollout_params_to_model(model: mujoco.MjModel, rollout_params: Any) -> None:
    _remove_mujoco_robot_limits(model)
    model.dof_frictionloss[:] = 0.0

    for field in ("dof_armature", "dof_damping"):
        arr = _as_np_field(getattr(rollout_params, field, None))
        if arr is not None:
            fitted = _fit_prefix(arr, model.__getattribute__(field))
            model.__getattribute__(field)[: fitted.shape[0]] = fitted

    for field in ("body_mass", "body_inertia", "body_ipos"):
        arr = _as_np_field(getattr(rollout_params, field, None))
        if arr is not None:
            target = model.__getattribute__(field)
            axis = -2 if target.ndim == 2 else -1
            fitted = _fit_prefix(arr, target, axis=axis)
            if fitted is not None:
                target[: fitted.shape[0]] = fitted


def _directional_param(param: Optional[np.ndarray], signal: np.ndarray) -> Optional[np.ndarray]:
    if param is None:
        return None
    param = np.asarray(param, dtype=np.float64)
    if param.ndim >= 2 and param.shape[-1] == 2:
        return np.where(signal >= 0.0, param[..., 1], param[..., 0])
    return param


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _actuator_model_np(tau: np.ndarray, q: np.ndarray, dq: np.ndarray, rollout_params: Any) -> np.ndarray:
    del q
    tau = np.asarray(tau, dtype=np.float64).copy()
    dq = np.asarray(dq, dtype=np.float64)

    torque_scale = _as_np_field(getattr(rollout_params, "torque_scale", None))
    if torque_scale is not None:
        tau = tau * np.where(tau >= 0.0, torque_scale[..., 1], torque_scale[..., 0])

    deadzone = _as_np_field(getattr(rollout_params, "deadzone", None))
    if deadzone is not None:
        dz_neg = deadzone[..., 0] if deadzone.ndim >= 2 and deadzone.shape[-1] == 2 else deadzone
        dz_pos = deadzone[..., 1] if deadzone.ndim >= 2 and deadzone.shape[-1] == 2 else deadzone
        abs_tau = np.abs(tau)
        sign_tau = np.sign(tau)
        dz_dir = np.where(sign_tau >= 0.0, dz_pos, dz_neg)
        tau = np.where(
            abs_tau <= dz_dir,
            tau * DEADZONE_SLOPE,
            sign_tau * (abs_tau - dz_dir + DEADZONE_SLOPE * dz_dir),
        )

    torque_bias = _as_np_field(getattr(rollout_params, "torque_bias", None))
    if torque_bias is not None:
        tau = tau + _directional_param(torque_bias, dq)

    damping = _as_np_field(getattr(rollout_params, "damping", None))
    if damping is not None:
        tau = tau - _directional_param(damping, dq) * dq

    friction_params = _as_np_field(getattr(rollout_params, "friction_params", None))
    if friction_params is not None:
        amp = friction_params[..., 0:2]
        slope = friction_params[..., 2:4]
        shift = friction_params[..., 4:6]
        amp_dir = _directional_param(amp, dq)
        slope_dir = _directional_param(slope, dq)
        shift_dir = _directional_param(shift, dq)
        tau = tau - amp_dir * (_sigmoid(slope_dir * (dq + shift_dir)) - _sigmoid(slope_dir * shift_dir))

    return tau


def _clip_to_actuator_range(model: mujoco.MjModel, tau: np.ndarray) -> np.ndarray:
    del model
    return np.asarray(tau, dtype=np.float64)


def _push_short(window: np.ndarray, value: np.ndarray) -> np.ndarray:
    out = np.empty_like(window)
    out[:-1] = window[1:]
    out[-1] = np.asarray(value, dtype=window.dtype)
    return out


def _maybe_update_tam(
    tam: OnlineTamSimState,
    *,
    timestamps: list[float],
    q_rows: list[np.ndarray],
    dq_rows: list[np.ndarray],
    tau_rows: list[np.ndarray],
    tau_base_rows: Optional[list[np.ndarray]] = None,
    tau_delta_rows: Optional[list[np.ndarray]] = None,
) -> None:
    if len(timestamps) < 2:
        return
    ts = np.asarray(timestamps, dtype=np.float64)
    q = np.asarray(q_rows, dtype=np.float32)
    dq = np.asarray(dq_rows, dtype=np.float32)
    tau_applied = np.asarray(tau_rows, dtype=np.float32)
    if tam.uses_fused_history:
        if tau_base_rows is None or tau_delta_rows is None:
            raise ValueError(
                "base_tam_fusion history requires tau_base_rows and tau_delta_rows."
            )
        # All three streams share the applied-torque validity mask so that
        # legitimately-zero base/residual rows are not dropped.
        applied_emb = tam.runtime.push_window(
            timestamps=ts,
            q=q,
            qd=dq,
            tau=tau_applied,
            tau_is_model_space=True,
            raw_tau=tau_applied,
        )
        base_emb = tam.base_runtime.push_window(
            timestamps=ts,
            q=q,
            qd=dq,
            tau=np.asarray(tau_base_rows, dtype=np.float32),
            tau_is_model_space=True,
            raw_tau=tau_applied,
        )
        tam_emb = tam.tam_runtime.push_window(
            timestamps=ts,
            q=q,
            qd=dq,
            tau=np.asarray(tau_delta_rows, dtype=np.float32),
            tau_is_model_space=True,
            raw_tau=tau_applied,
        )
        emitted = [applied_emb is not None, base_emb is not None, tam_emb is not None]
        if any(emitted) and not all(emitted):
            raise RuntimeError(
                "Fused history streams emitted embeddings out of lockstep "
                f"(applied/base/tam={emitted}); identical timestamps must yield "
                "identical patch cadence."
            )
        if not all(emitted):
            return
        from simadaptor.deploy.mapping_server import _apply_history_fusion

        emb = _apply_history_fusion(
            tam.history_fusion_params,
            applied_emb,
            base_emb,
            tam_emb,
        )
    else:
        emb = tam.runtime.push_window(
            timestamps=ts,
            q=q,
            qd=dq,
            tau=tau_applied,
            tau_is_model_space=True,
        )
    if emb is None:
        return
    tam.num_embeddings += 1
    tam.patches_since_reset += 1
    if tam.patches_since_reset < int(tam.min_patches_before_send):
        return
    t_now = float(timestamps[-1])
    if (t_now - float(tam.last_send_t)) < float(tam.embedding_interval_s):
        return
    tam.current_embedding = emb
    tam.num_sent += 1
    if tam.delay_enable and not tam.enabled:
        tam.enabled = True
    tam.last_send_t = t_now


def _apply_tam_if_available(
    tam: Optional[OnlineTamSimState],
    *,
    q_window: np.ndarray,
    dq_window: np.ndarray,
    tau_plain_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tau_plain = np.asarray(tau_plain_window[-1], dtype=np.float32)
    if tam is None or not tam.enabled or tam.current_embedding is None:
        return tau_plain.astype(np.float64), np.zeros_like(tau_plain, dtype=np.float64)

    import jax
    import jax.numpy as jnp

    delta_tau, _ = tam.adaptor_apply_jit(
        tam.model_params["adaptor"],
        jnp.asarray(q_window, dtype=jnp.float32)[None, ...],
        jnp.asarray(dq_window, dtype=jnp.float32)[None, ...],
        jnp.asarray(tau_plain_window, dtype=jnp.float32)[None, ...],
        tam.current_embedding,
        jax.random.PRNGKey(0),
        False,
        tam.norm_stats,
    )
    delta = np.asarray(delta_tau[0], dtype=np.float64)
    return np.asarray(tau_plain, dtype=np.float64) + delta, delta


def _empty_log() -> dict[str, list[Any]]:
    return {
        "t": [],
        "q": [],
        "dq": [],
        "target_q": [],
        "target_dq": [],
        "target_pos": [],
        "target_quat": [],
        "ee_pos": [],
        "ee_quat": [],
        "tau_plain": [],
        "tau_cmd": [],
        "tau_delta": [],
        "tam_enabled": [],
    }


def _append_log(
    log: dict[str, list[Any]],
    *,
    t: float,
    q: np.ndarray,
    dq: np.ndarray,
    tau_plain: np.ndarray,
    tau_cmd: np.ndarray,
    tau_delta: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    tam_enabled: bool,
    target_q: Optional[np.ndarray] = None,
    target_dq: Optional[np.ndarray] = None,
    target_pos: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
) -> None:
    log["t"].append(float(t))
    log["q"].append(np.asarray(q, dtype=np.float32).copy())
    log["dq"].append(np.asarray(dq, dtype=np.float32).copy())
    log["tau_plain"].append(np.asarray(tau_plain, dtype=np.float32).copy())
    log["tau_cmd"].append(np.asarray(tau_cmd, dtype=np.float32).copy())
    log["tau_delta"].append(np.asarray(tau_delta, dtype=np.float32).copy())
    log["ee_pos"].append(np.asarray(ee_pos, dtype=np.float32).copy())
    log["ee_quat"].append(np.asarray(ee_quat, dtype=np.float32).copy())
    log["tam_enabled"].append(bool(tam_enabled))
    if target_q is not None:
        log["target_q"].append(np.asarray(target_q, dtype=np.float32).copy())
    if target_dq is not None:
        log["target_dq"].append(np.asarray(target_dq, dtype=np.float32).copy())
    if target_pos is not None:
        log["target_pos"].append(np.asarray(target_pos, dtype=np.float32).copy())
    if target_quat is not None:
        log["target_quat"].append(np.asarray(target_quat, dtype=np.float32).copy())


def _log_to_np(log: dict[str, list[Any]], *, dof: int = 7) -> dict[str, np.ndarray]:
    shapes = {
        "q": (int(dof),),
        "dq": (int(dof),),
        "target_q": (int(dof),),
        "target_dq": (int(dof),),
        "target_pos": (3,),
        "target_quat": (4,),
        "ee_pos": (3,),
        "ee_quat": (4,),
        "tau_plain": (int(dof),),
        "tau_cmd": (int(dof),),
        "tau_delta": (int(dof),),
    }
    out: dict[str, np.ndarray] = {}
    for key, values in log.items():
        if key == "tam_enabled":
            out[key] = np.asarray(values, dtype=np.bool_)
        elif key == "t":
            out[key] = np.asarray(values, dtype=np.float64)
        elif values:
            out[key] = np.asarray(values)
        else:
            out[key] = np.empty((0, *shapes.get(key, ())), dtype=np.float32)
    return out


def _target_metrics_from_np(
    *,
    target_log_np: dict[str, np.ndarray],
    target_t: np.ndarray,
    target_pos: np.ndarray,
) -> dict[str, float | int]:
    proxy_log = {
        "ee_pos": [row for row in np.asarray(target_log_np.get("ee_pos", np.empty((0, 3))))],
        "ee_pos_t": [float(x) for x in np.asarray(target_log_np.get("t", np.empty((0,))))],
    }
    return compute_target_metrics(
        target_log=proxy_log,
        t_target=np.asarray(target_t, dtype=np.float64),
        pos_target=np.asarray(target_pos, dtype=np.float64),
    )


def _trajectory_error_metrics(
    *,
    target_log_np: dict[str, np.ndarray],
    ideal_log_np: dict[str, np.ndarray],
) -> dict[str, float | int]:
    actual_log = {
        "ee_pos": [row for row in np.asarray(target_log_np.get("ee_pos", np.empty((0, 3))), dtype=np.float64)],
        "ee_pos_t": [float(x) for x in np.asarray(target_log_np.get("t", np.empty((0,))), dtype=np.float64)],
    }
    ideal_t = np.asarray(ideal_log_np.get("t", np.empty((0,))), dtype=np.float64).reshape(-1)
    ideal_pos = np.asarray(ideal_log_np.get("ee_pos", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    ideal_quat = np.asarray(ideal_log_np.get("ee_quat", np.empty((0, 4))), dtype=np.float64).reshape(-1, 4)
    if ideal_t.shape[0] != ideal_pos.shape[0] or ideal_t.shape[0] < 2:
        return {
            "target_ideal_ee_samples": int(ideal_pos.shape[0]),
            "target_vs_ideal_ee_pos_rmse_m": float("nan"),
            "target_vs_ideal_ee_final_error_m": float("nan"),
            "target_vs_ideal_ee_ori_rmse_deg": float("nan"),
            "target_vs_ideal_ee_final_ori_error_deg": float("nan"),
        }
    metrics = compute_target_metrics(
        target_log=actual_log,
        t_target=ideal_t,
        pos_target=ideal_pos,
        quat_target=ideal_quat if ideal_quat.shape[0] == ideal_t.shape[0] else None,
    )
    return {
        "target_ideal_ee_samples": int(ideal_pos.shape[0]),
        "target_vs_ideal_ee_pos_samples": metrics["target_vs_reference_ee_pos_samples"],
        "target_vs_ideal_ee_pos_rmse_m": metrics["target_vs_reference_ee_pos_rmse_m"],
        "target_vs_ideal_ee_final_error_m": metrics["target_vs_reference_ee_final_error_m"],
        "target_vs_ideal_ee_ori_samples": metrics.get("target_vs_reference_ee_ori_samples", 0),
        "target_vs_ideal_ee_ori_rmse_deg": metrics.get("target_vs_reference_ee_ori_rmse_deg", float("nan")),
        "target_vs_ideal_ee_final_ori_error_deg": metrics.get(
            "target_vs_reference_ee_final_ori_error_deg",
            float("nan"),
        ),
    }


def _interp_columns(*, source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    values = np.asarray(source_values, dtype=np.float64)
    if source_t.shape[0] != values.shape[0] or source_t.shape[0] < 2:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float64)
    values_2d = values.reshape(values.shape[0], -1)
    return np.stack(
        [np.interp(target_t, source_t, values_2d[:, col]) for col in range(values_2d.shape[1])],
        axis=1,
    )


def _source_vs_ideal_metrics(
    *,
    source_log_np: dict[str, np.ndarray],
    ideal_source_log_np: Optional[dict[str, np.ndarray]],
) -> dict[str, float | int]:
    source_t = np.asarray(source_log_np.get("t", np.empty((0,))), dtype=np.float64).reshape(-1)
    source_q = np.asarray(source_log_np.get("q", np.empty((0, 0))), dtype=np.float64)
    source_ee = np.asarray(source_log_np.get("ee_pos", np.empty((0, 3))), dtype=np.float64)
    ideal_t = (
        np.empty((0,), dtype=np.float64)
        if ideal_source_log_np is None
        else np.asarray(ideal_source_log_np.get("t", np.empty((0,))), dtype=np.float64).reshape(-1)
    )
    ideal_q = (
        np.empty((0, source_q.shape[1] if source_q.ndim == 2 else 0), dtype=np.float64)
        if ideal_source_log_np is None
        else np.asarray(ideal_source_log_np.get("q", np.empty((0, 0))), dtype=np.float64)
    )
    ideal_ee = (
        np.empty((0, 3), dtype=np.float64)
        if ideal_source_log_np is None
        else np.asarray(ideal_source_log_np.get("ee_pos", np.empty((0, 3))), dtype=np.float64)
    )
    empty = {
        "source_ideal_joint_samples": int(ideal_q.shape[0]),
        "source_vs_ideal_joint_pos_samples": 0,
        "source_vs_ideal_joint_pos_rmse_rad": float("nan"),
        "source_vs_ideal_joint_pos_rmse_deg": float("nan"),
        "source_vs_ideal_joint_final_error_rad": float("nan"),
        "source_ideal_ee_samples": int(ideal_ee.shape[0]),
        "source_vs_ideal_ee_pos_samples": 0,
        "source_vs_ideal_ee_pos_rmse_m": float("nan"),
        "source_vs_ideal_ee_final_error_m": float("nan"),
    }
    if source_t.shape[0] < 2 or ideal_t.shape[0] < 2 or source_q.shape[0] != source_t.shape[0]:
        return empty
    ideal_q_at_source = _interp_columns(source_t=ideal_t, source_values=ideal_q, target_t=source_t)
    if ideal_q_at_source.shape != source_q.reshape(source_q.shape[0], -1).shape:
        return empty
    q_err = source_q.reshape(source_q.shape[0], -1) - ideal_q_at_source
    out = {
        **empty,
        "source_vs_ideal_joint_pos_samples": int(source_t.shape[0]),
        "source_vs_ideal_joint_pos_rmse_rad": float(np.sqrt(np.mean(q_err * q_err))),
        "source_vs_ideal_joint_pos_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(q_err * q_err)))),
        "source_vs_ideal_joint_final_error_rad": float(np.linalg.norm(q_err[-1])),
    }
    if source_ee.shape[0] == source_t.shape[0] and ideal_ee.shape[0] == ideal_t.shape[0]:
        ideal_ee_at_source = _interp_columns(source_t=ideal_t, source_values=ideal_ee, target_t=source_t)
        if ideal_ee_at_source.shape == source_ee.reshape(source_ee.shape[0], -1).shape:
            ee_err = source_ee.reshape(source_ee.shape[0], -1) - ideal_ee_at_source
            out.update(
                {
                    "source_vs_ideal_ee_pos_samples": int(source_t.shape[0]),
                    "source_vs_ideal_ee_pos_rmse_m": float(
                        np.sqrt(np.mean(np.sum(ee_err * ee_err, axis=-1)))
                    ),
                    "source_vs_ideal_ee_final_error_m": float(np.linalg.norm(ee_err[-1])),
                }
            )
    return out


def _write_summary_files(run_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "iteration",
        "condition_key",
        "condition",
        "source_segment",
        "switch_behavior",
        "target_osc_segment",
        "source_seed",
        "sim_seed",
        "osc_delta_xyz",
        "target_ee_samples",
        "target_ee_pos_rmse_m",
        "target_ee_final_error_m",
        "target_ideal_ee_samples",
        "target_vs_ideal_ee_pos_rmse_m",
        "target_vs_ideal_ee_final_error_m",
        "source_ideal_joint_samples",
        "source_vs_ideal_joint_pos_rmse_deg",
        "source_vs_ideal_joint_final_error_rad",
        "source_ideal_ee_samples",
        "source_vs_ideal_ee_pos_rmse_m",
        "source_vs_ideal_ee_final_error_m",
        "source_final_q_error_rad",
        "switch_actual_to_osc_start_pos_error_m",
        "controller_side_guard",
        "controller_guard_velocity_threshold",
        "tam_embeddings_total",
        "tam_embeddings_sent",
    ]
    (run_dir / "summary.json").write_text(
        json.dumps(list(rows), indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "| Iter | Condition | Source segment | Switch behavior | Target OSC segment | Cmd RMSE (m) | Ideal RMSE (m) | Final vs ideal (m) | TAM sent |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cmd_rmse = row.get("target_ee_pos_rmse_m", float("nan"))
        ideal_rmse = row.get("target_vs_ideal_ee_pos_rmse_m", float("nan"))
        ideal_final = row.get("target_vs_ideal_ee_final_error_m", float("nan"))
        lines.append(
            "| {iteration} | {condition} | {source_segment} | {switch_behavior} | {target} | {cmd_rmse} | {ideal_rmse} | {ideal_final} | {sent} |".format(
                iteration=int(row.get("iteration", 0) or 0),
                condition=row.get("condition", ""),
                source_segment=row.get("source_segment", ""),
                switch_behavior=row.get("switch_behavior", ""),
                target=row.get("target_osc_segment", ""),
                cmd_rmse="" if not np.isfinite(cmd_rmse) else f"{float(cmd_rmse):.6f}",
                ideal_rmse="" if not np.isfinite(ideal_rmse) else f"{float(ideal_rmse):.6f}",
                ideal_final="" if not np.isfinite(ideal_final) else f"{float(ideal_final):.6f}",
                sent=int(row.get("tam_embeddings_sent", 0) or 0),
            )
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite_stats(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def _write_aggregate_files(run_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition_key", "")), []).append(dict(row))

    metric_keys = [
        "target_ee_pos_rmse_m",
        "target_ee_final_error_m",
        "target_vs_ideal_ee_pos_rmse_m",
        "target_vs_ideal_ee_final_error_m",
        "source_vs_ideal_joint_pos_rmse_deg",
        "source_vs_ideal_joint_final_error_rad",
        "source_vs_ideal_ee_pos_rmse_m",
        "source_vs_ideal_ee_final_error_m",
        "source_final_q_error_rad",
        "switch_actual_to_osc_start_pos_error_m",
        "tam_embeddings_total",
        "tam_embeddings_sent",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for condition_key, condition_rows in grouped.items():
        first = condition_rows[0]
        out: dict[str, Any] = {
            "condition_key": condition_key,
            "condition": first.get("condition", ""),
            "iterations": len(condition_rows),
            "source_segment": first.get("source_segment", ""),
            "switch_behavior": first.get("switch_behavior", ""),
            "target_osc_segment": first.get("target_osc_segment", ""),
        }
        for metric_key in metric_keys:
            stats = _finite_stats(condition_rows, metric_key)
            out[f"{metric_key}_n"] = stats["n"]
            out[f"{metric_key}_mean"] = stats["mean"]
            out[f"{metric_key}_std"] = stats["std"]
        aggregate_rows.append(out)

    fieldnames = [
        "condition_key",
        "condition",
        "iterations",
        "source_segment",
        "switch_behavior",
        "target_osc_segment",
    ]
    for metric_key in metric_keys:
        fieldnames.extend([f"{metric_key}_n", f"{metric_key}_mean", f"{metric_key}_std"])
    fieldnames.extend(
        [
            "target_vs_ideal_ee_pos_rmse_mm_mean",
            "target_vs_ideal_ee_pos_rmse_mm_std",
            "target_vs_ideal_ee_pos_rmse_mm_pm",
            "source_vs_ideal_ee_pos_rmse_mm_mean",
            "source_vs_ideal_ee_pos_rmse_mm_std",
            "source_vs_ideal_ee_pos_rmse_mm_pm",
            "source_vs_ideal_joint_pos_rmse_deg_pm",
        ]
    )
    for row in aggregate_rows:
        mean_m = row.get("target_vs_ideal_ee_pos_rmse_m_mean", float("nan"))
        std_m = row.get("target_vs_ideal_ee_pos_rmse_m_std", float("nan"))
        row["target_vs_ideal_ee_pos_rmse_mm_mean"] = (
            float(mean_m) * 1000.0 if np.isfinite(mean_m) else float("nan")
        )
        row["target_vs_ideal_ee_pos_rmse_mm_std"] = (
            float(std_m) * 1000.0 if np.isfinite(std_m) else float("nan")
        )
        row["target_vs_ideal_ee_pos_rmse_mm_pm"] = (
            ""
            if not (
                np.isfinite(row["target_vs_ideal_ee_pos_rmse_mm_mean"])
                and np.isfinite(row["target_vs_ideal_ee_pos_rmse_mm_std"])
            )
            else f"{row['target_vs_ideal_ee_pos_rmse_mm_mean']:.2f} +/- {row['target_vs_ideal_ee_pos_rmse_mm_std']:.2f}"
        )
        mean_m = row.get("source_vs_ideal_ee_pos_rmse_m_mean", float("nan"))
        std_m = row.get("source_vs_ideal_ee_pos_rmse_m_std", float("nan"))
        row["source_vs_ideal_ee_pos_rmse_mm_mean"] = (
            float(mean_m) * 1000.0 if np.isfinite(mean_m) else float("nan")
        )
        row["source_vs_ideal_ee_pos_rmse_mm_std"] = (
            float(std_m) * 1000.0 if np.isfinite(std_m) else float("nan")
        )
        row["source_vs_ideal_ee_pos_rmse_mm_pm"] = (
            ""
            if not (
                np.isfinite(row["source_vs_ideal_ee_pos_rmse_mm_mean"])
                and np.isfinite(row["source_vs_ideal_ee_pos_rmse_mm_std"])
            )
            else f"{row['source_vs_ideal_ee_pos_rmse_mm_mean']:.2f} +/- {row['source_vs_ideal_ee_pos_rmse_mm_std']:.2f}"
        )
        mean_deg = row.get("source_vs_ideal_joint_pos_rmse_deg_mean", float("nan"))
        std_deg = row.get("source_vs_ideal_joint_pos_rmse_deg_std", float("nan"))
        row["source_vs_ideal_joint_pos_rmse_deg_pm"] = (
            ""
            if not (np.isfinite(mean_deg) and np.isfinite(std_deg))
            else f"{float(mean_deg):.2f} +/- {float(std_deg):.2f}"
        )
    (run_dir / "summary_aggregate.json").write_text(
        json.dumps(aggregate_rows, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    with (run_dir / "summary_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "| Condition | Iter | Source EE RMSE vs ideal (mm) | Source q RMSE vs ideal (deg) | OSC pos RMSE vs ideal (mm) | Cmd RMSE mean (m) | Final vs ideal mean (m) | TAM sent mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        def fmt(key: str) -> str:
            value = row.get(key, float("nan"))
            return "" if not np.isfinite(value) else f"{float(value):.6f}"

        lines.append(
            "| {condition} | {iterations} | {source_ee_mm} | {source_q_deg} | {ideal_mm} | {cmd_mean} | {ideal_final_mean} | {tam_mean} |".format(
                condition=row.get("condition", ""),
                iterations=int(row.get("iterations", 0) or 0),
                source_ee_mm=row.get("source_vs_ideal_ee_pos_rmse_mm_pm", ""),
                source_q_deg=row.get("source_vs_ideal_joint_pos_rmse_deg_pm", ""),
                ideal_mm=row.get("target_vs_ideal_ee_pos_rmse_mm_pm", ""),
                cmd_mean=fmt("target_ee_pos_rmse_m_mean"),
                ideal_final_mean=fmt("target_vs_ideal_ee_final_error_m_mean"),
                tam_mean=fmt("tam_embeddings_sent_mean"),
            )
        )
    (run_dir / "summary_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_profile(
    *,
    xml_path: Path,
    profile_table: Path,
    profile_key: Optional[str],
) -> tuple[str, dict[str, Any]]:
    from simadaptor.data.datagen_profiles import derive_robot_key, load_datagen_profile

    robot_key = derive_robot_key(xml_path)
    key = profile_key
    if key is None and robot_key == "robot":
        key = DEFAULT_PROFILE_KEY
    resolved_key, kwargs = load_datagen_profile(
        table_path=profile_table,
        robot_key=robot_key,
        profile_key=key,
    )
    return resolved_key, kwargs


def _sample_params(
    *,
    xml_path: Path,
    profile_table: Path,
    profile_key: Optional[str],
    seed: int,
    dt_s: float,
    param_debug_mode: str = "default",
) -> tuple[Any, Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    import simadaptor.core.structs as structs
    import simadaptor.data.datagen as datagen
    import simadaptor.physics.dynamics as dynamics

    mjx_model = dynamics.load_mjx_model_from_path(str(xml_path), remove_constraints=True)
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(timestep=float(dt_s)))
    resolved_profile_key, profile_kwargs = _resolve_profile(
        xml_path=xml_path,
        profile_table=profile_table,
        profile_key=profile_key,
    )
    sample_keys = set(inspect.signature(datagen.sample_random_params).parameters.keys())
    sample_kwargs = {k: v for k, v in profile_kwargs.items() if k in sample_keys}
    dof = int(mjx_model.nu)
    if int(np.asarray(profile_kwargs["base_kp_profile"]).shape[0]) != dof:
        raise ValueError(
            f"Profile '{resolved_profile_key}' dof does not match model.nu={dof}."
        )

    base_kp_profile = jnp.asarray(profile_kwargs["base_kp_profile"], dtype=jnp.float32)
    dof_mass_diag_q0 = dynamics.get_mass_matrix_diag_at_qpos0(mjx_model)
    actuator_mass_diag_q0 = datagen._actuator_mass_diag_from_model(mjx_model, dof_mass_diag_q0)
    kd_scale_range = sample_kwargs.get("kd_scale_range", None)
    kd_lo, kd_hi = (0.5, 1.5) if kd_scale_range is None else (float(kd_scale_range[0]), float(kd_scale_range[1]))
    if kd_hi < kd_lo:
        kd_lo, kd_hi = kd_hi, kd_lo

    rng = jax.random.PRNGKey(int(seed))
    rng_nominal, rng_perturbed, rng_kd = jax.random.split(rng, 3)
    nominal_dict, _, _ = datagen.sample_random_params(
        rng_nominal,
        mjx_model,
        evaluation_mode=True,
        **sample_kwargs,
    )
    _, perturbed_dict, _ = datagen.sample_random_params(
        rng_perturbed,
        mjx_model,
        evaluation_mode=True,
        **sample_kwargs,
    )

    nominal_dict["kp"] = base_kp_profile * 0.2
    random_kd_scale = jax.random.uniform(
        rng_kd,
        (dof,),
        minval=kd_lo,
        maxval=kd_hi,
    )
    nominal_dict["kd"] = random_kd_scale * 2.0 * jnp.sqrt(
        jnp.maximum(nominal_dict["kp"], 1e-6) * jnp.maximum(actuator_mass_diag_q0, 1e-6)
    )
    ideal_params = structs.RolloutParams(**nominal_dict)
    perturbed_params = structs.RolloutParams(**perturbed_dict)
    if str(param_debug_mode) == "actuator_major_no_payload":
        # Keep the sampled actuator/controller perturbations from this seed, but
        # remove body/payload randomization so the diagnostic isolates actuator-side mismatch.
        perturbed_params = perturbed_params.replace(
            body_mass=ideal_params.body_mass,
            body_mass_delta=ideal_params.body_mass_delta,
            body_inertia=ideal_params.body_inertia,
            body_ipos=ideal_params.body_ipos,
        )
    elif str(param_debug_mode) != "default":
        raise ValueError(f"Unsupported param_debug_mode={param_debug_mode!r}.")
    return ideal_params, perturbed_params, {
        "profile_key": resolved_profile_key,
        "physics_rng_seed": int(seed),
        "param_debug_mode": str(param_debug_mode),
        "mjx_model_nu": int(mjx_model.nu),
        "ideal_param_summary": _rollout_param_summary(ideal_params),
        "perturbed_param_summary": _rollout_param_summary(perturbed_params),
    }


@dataclasses.dataclass(frozen=True)
class FastTamBundle:
    hist_model: Any
    adaptor_model: Any
    adaptor_apply_fn: Any
    model_params: Any
    norm_stats: Any
    cfg: Any
    ablation_mode: str
    adaptor_seq_length: int
    emb_dim: int


def _load_fast_tam_bundle(args: argparse.Namespace) -> FastTamBundle:
    _install_optional_viser_stub()
    from simadaptor.deploy.inf_util import SimAdaptorInference

    ckpt_args = [
        args.adaptor_ckpt_path is not None,
    ]
    if sum(1 for x in ckpt_args if x) != 1:
        raise SystemExit(
            "Batched source-to-OSC TAM rows need --tam-ckpt-path."
        )
    sim_inf = SimAdaptorInference(
        simadaptor_ckpt_path=str(args.adaptor_ckpt_path) if args.adaptor_ckpt_path is not None else None,
        xml_path=args.xml,
        load_collision_models=False,
    )
    args.resolved_history_torque_mode = _require_applied_history_torque_mode(sim_inf)
    _history_apply_fn, adaptor_apply_jit, model_params, norm_stats, cfg = sim_inf.get_apply_fns()
    del _history_apply_fn
    hist_model, adaptor_model = getattr(sim_inf, "_simadaptor_model")
    ablation_mode = str(getattr(cfg, "ablation_mode", "tam") or "tam")
    return FastTamBundle(
        hist_model=hist_model,
        adaptor_model=adaptor_model,
        adaptor_apply_fn=adaptor_apply_jit,
        model_params=model_params,
        norm_stats=norm_stats,
        cfg=cfg,
        ablation_mode=ablation_mode,
        adaptor_seq_length=max(int(getattr(cfg, "adaptor_seq_length", 1) or 1), 1),
        emb_dim=int(getattr(cfg, "emb_dim")),
    )


def _resolve_history_torque_mode(adaptor_or_inf: Any) -> str:
    """Resolve a checkpoint's history torque mode, failing on unknown modes."""

    # Mirror mapping_server._simadaptor_config_history_torque_mode without
    # importing the full deploy server into this simulator. DAgger metadata is
    # authoritative, then the saved config, then fusion-weight auto-detection.
    inf = getattr(adaptor_or_inf, "inf", None) or adaptor_or_inf
    dagger_cfg = getattr(inf, "dagger_cfg", None)
    cfg = getattr(inf, "cfg", None)
    configured = getattr(dagger_cfg, "history_torque_mode", None)
    if not configured:
        configured = getattr(cfg, "history_torque_mode", None)
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        has_history_fusion = "history_fusion" in params
    except Exception:
        has_history_fusion = False
    resolved = (
        str(configured).strip()
        if configured
        else ("base_tam_fusion" if has_history_fusion else "applied")
    )
    if resolved not in {"applied", "base_tam_fusion"}:
        raise SystemExit(
            f"Unsupported checkpoint history_torque_mode={resolved!r}; this "
            "simulator implements 'applied' and 'base_tam_fusion'."
        )
    if resolved == "base_tam_fusion" and not has_history_fusion:
        raise SystemExit(
            "Checkpoint requests base_tam_fusion history but has no "
            "params['history_fusion'] weights."
        )
    return resolved


def _history_fusion_params_from_inf(inf: Any) -> Any:
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        return params["history_fusion"]
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint requested base_tam_fusion history, but history_fusion "
            "parameters are missing."
        ) from exc


def _require_applied_history_torque_mode(adaptor_or_inf: Any) -> str:
    """Fail closed when a checkpoint requires history streams this path lacks."""

    resolved = _resolve_history_torque_mode(adaptor_or_inf)
    if resolved != "applied":
        raise SystemExit(
            "The batched source-to-OSC backend supports only the single "
            "applied-torque history stream, but the checkpoint resolves to "
            f"history_torque_mode={resolved!r}. Use --sim-backend legacy for "
            "fused base/TAM history checkpoints."
        )
    return str(resolved)


def _sim_backend_choice(args: argparse.Namespace, conditions: Sequence[SimConditionSpec]) -> str:
    requested = str(getattr(args, "sim_backend", "auto") or "auto").lower()
    if requested not in {"auto", "legacy", "batched"}:
        raise SystemExit(f"Unknown --sim-backend={requested!r}; expected auto, legacy, or batched.")
    source_only = bool(getattr(args, "source_only", False))
    if source_only and requested == "batched":
        raise SystemExit(
            "--source-only is not supported by --sim-backend batched; use "
            "--sim-backend legacy."
        )
    if requested != "auto":
        return requested
    if source_only:
        return "legacy"
    keys = {spec.key for spec in conditions}
    table_keys = {"direct_osc", "tam_carried"}
    if keys and keys.issubset(table_keys):
        return "batched"
    return "legacy"


def _stack_rollout_params(params_list: Sequence[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda *xs: None if xs[0] is None else jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *params_list,
    )


def _sample_params_for_references(
    *,
    args: argparse.Namespace,
    refs: Sequence[SimReference],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    ideal_params_list = []
    perturbed_params_list = []
    setup_rows: list[dict[str, Any]] = []
    for local_iteration, ref in enumerate(refs):
        ideal_params, perturbed_params, setup_meta = _sample_params(
            xml_path=args.xml.expanduser().resolve(),
            profile_table=args.profile_table.expanduser().resolve(),
            profile_key=args.profile_key,
            seed=int(ref.sim_seed),
            dt_s=float(args.dt),
            param_debug_mode=str(args.param_debug_mode),
        )
        ideal_params_list.append(ideal_params)
        perturbed_params_list.append(perturbed_params)
        setup_rows.append(
            {
                "local_iteration": int(local_iteration),
                "iteration": int(ref.iteration),
                "source_seed": int(ref.source_seed),
                "sim_seed": int(ref.sim_seed),
                "osc_delta_xyz": list(ref.osc_delta_xyz),
                "osc_delta_rpy_deg": list(ref.osc_delta_rpy_deg),
                "osc_waypoint_xyz": [list(row) for row in ref.osc_waypoint_xyz],
                "osc_waypoint_rpy_deg": [list(row) for row in ref.osc_waypoint_rpy_deg],
                "source_amp_deg": list(ref.source_amp_deg),
                "source_cycles": list(ref.source_cycles),
                "resolved_history_torque_mode": str(
                    getattr(args, "resolved_history_torque_mode", "") or ""
                ),
                **setup_meta,
            }
        )
    return _stack_rollout_params(ideal_params_list), _stack_rollout_params(perturbed_params_list), setup_rows


def _quat_wxyz_conj_jax(q):
    import jax.numpy as jnp

    q = jnp.asarray(q, dtype=jnp.float32)
    return jnp.concatenate([q[:1], -q[1:]], axis=0)


def _quat_wxyz_mul_jax(a, b):
    import jax.numpy as jnp

    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    q = jnp.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=jnp.float32,
    )
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def _batched_target_metrics_np(
    *,
    target_ee: np.ndarray,
    target_ref: np.ndarray,
    ideal_ee: np.ndarray,
) -> dict[str, np.ndarray]:
    target_ee = np.asarray(target_ee, dtype=np.float64)
    target_ref = np.asarray(target_ref, dtype=np.float64)
    ideal_ee = np.asarray(ideal_ee, dtype=np.float64)
    cmd_err = target_ee - target_ref
    ideal_err = target_ee - ideal_ee
    return {
        "target_ee_pos_rmse_m": np.sqrt(np.mean(np.sum(cmd_err * cmd_err, axis=-1), axis=-1)),
        "target_ee_final_error_m": np.linalg.norm(cmd_err[:, -1, :], axis=-1),
        "target_vs_ideal_ee_pos_rmse_m": np.sqrt(
            np.mean(np.sum(ideal_err * ideal_err, axis=-1), axis=-1)
        ),
        "target_vs_ideal_ee_final_error_m": np.linalg.norm(ideal_err[:, -1, :], axis=-1),
    }


def _batched_source_vs_ideal_metrics_np(
    *,
    source_q: np.ndarray,
    source_ee: np.ndarray,
    ideal_source_q: np.ndarray,
    ideal_source_ee: np.ndarray,
) -> dict[str, np.ndarray]:
    source_q = np.asarray(source_q, dtype=np.float64)
    source_ee = np.asarray(source_ee, dtype=np.float64)
    ideal_source_q = np.asarray(ideal_source_q, dtype=np.float64)
    ideal_source_ee = np.asarray(ideal_source_ee, dtype=np.float64)
    q_err = source_q - ideal_source_q
    ee_err = source_ee - ideal_source_ee
    joint_rmse_rad = np.sqrt(np.mean(q_err * q_err, axis=(-2, -1)))
    return {
        "source_ideal_joint_samples": np.full(source_q.shape[0], source_q.shape[1], dtype=np.int32),
        "source_vs_ideal_joint_pos_samples": np.full(source_q.shape[0], source_q.shape[1], dtype=np.int32),
        "source_vs_ideal_joint_pos_rmse_rad": joint_rmse_rad,
        "source_vs_ideal_joint_pos_rmse_deg": np.rad2deg(joint_rmse_rad),
        "source_vs_ideal_joint_final_error_rad": np.linalg.norm(q_err[:, -1, :], axis=-1),
        "source_ideal_ee_samples": np.full(source_ee.shape[0], source_ee.shape[1], dtype=np.int32),
        "source_vs_ideal_ee_pos_samples": np.full(source_ee.shape[0], source_ee.shape[1], dtype=np.int32),
        "source_vs_ideal_ee_pos_rmse_m": np.sqrt(np.mean(np.sum(ee_err * ee_err, axis=-1), axis=-1)),
        "source_vs_ideal_ee_final_error_m": np.linalg.norm(ee_err[:, -1, :], axis=-1),
    }


def _make_batched_source_to_osc_runner(
    *,
    args: argparse.Namespace,
    tam_bundle: Optional[FastTamBundle],
):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    import simadaptor.physics.actuator as actuator_util
    from simadaptor.core.transform_util import matrix_to_quat_wxyz
    from simadaptor.eval.online_runtime import (
        advance_online_history_state,
        apply_online_adaptor,
        build_online_history_runtime,
        init_online_history_state,
        push_window,
    )

    xml_path = args.xml.expanduser().resolve()
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_model.opt.timestep = float(args.dt)
    mj_model.body_gravcomp[:] = 0.0
    _remove_mujoco_robot_limits(mj_model)
    site_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, str(args.fk_site)))
    if site_id < 0:
        raise ValueError(f"Site not found in XML: {args.fk_site}")
    qpos_idx_np, qvel_idx_np, _joint_ids = _arm_indices(mj_model, dof=getattr(args, "arm_dof", None))
    mjx_model = mjx.put_model(mj_model)
    dof = int(len(qvel_idx_np))
    arm_qpos_ids = jnp.asarray(qpos_idx_np, dtype=jnp.int32)
    arm_qvel_ids = jnp.asarray(qvel_idx_np, dtype=jnp.int32)
    actuator_trnid = jnp.asarray(mjx_model.actuator_trnid, dtype=jnp.int32)
    jnt_dofadr = jnp.asarray(mjx_model.jnt_dofadr, dtype=jnp.int32)
    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = jnp.clip(act_jnt_id, 0, int(mjx_model.njnt) - 1)
    actuator_dof_abs = jnp.where(
        (act_jnt_id >= 0) & (act_jnt_id < int(mjx_model.njnt)),
        jnt_dofadr[act_jnt_id_clamped],
        jnp.minimum(jnp.arange(int(mjx_model.nu), dtype=jnp.int32), int(mjx_model.nv) - 1),
    )
    joint_kp = jnp.asarray(args.joint_stiffness, dtype=jnp.float32)
    if args.joint_damping is None:
        joint_kd = 2.0 * jnp.sqrt(joint_kp) * jnp.asarray(_default_damping_ratio(dof), dtype=jnp.float32)
    else:
        joint_kd = jnp.asarray(args.joint_damping, dtype=jnp.float32)
    osc_stiffness = jnp.asarray(args.osc_stiffness, dtype=jnp.float32)
    osc_damping = jnp.asarray(args.osc_damping, dtype=jnp.float32)
    osc_nullspace_stiffness = jnp.asarray(float(args.osc_nullspace_stiffness), dtype=jnp.float32)
    min_updates_before_use = max(int(args.min_patches_before_send), 1)
    controller_guard_enabled = bool(getattr(args, "controller_side_guard", True))
    controller_guard_velocity_threshold = jnp.asarray(
        float(getattr(args, "controller_guard_velocity_threshold", 4.0)),
        dtype=jnp.float32,
    )

    if tam_bundle is not None:
        runtime = build_online_history_runtime(
            hist_model=tam_bundle.hist_model,
            params_hist_example=tam_bundle.model_params["hist"],
            emb_dim=tam_bundle.emb_dim,
            arm_dof=dof,
        )
        adaptor_seq_length = int(tam_bundle.adaptor_seq_length)
        model_params = tam_bundle.model_params
        norm_stats = tam_bundle.norm_stats
        adaptor_model = tam_bundle.adaptor_model
        adaptor_apply_fn = tam_bundle.adaptor_apply_fn
    else:
        runtime = build_online_history_runtime(
            hist_model=None,
            params_hist_example=None,
            emb_dim=1,
            arm_dof=dof,
        )
        adaptor_seq_length = 1
        model_params = None
        norm_stats = None
        adaptor_model = None
        adaptor_apply_fn = None

    def _data_at(q, qd):
        data0 = mjx.make_data(mjx_model)
        return data0.replace(
            qpos=data0.qpos.at[arm_qpos_ids].set(jnp.asarray(q, dtype=jnp.float32)),
            qvel=data0.qvel.at[arm_qvel_ids].set(jnp.asarray(qd, dtype=jnp.float32)),
        )

    def _gravity_torque(q):
        data = mjx.forward(mjx_model, _data_at(q, jnp.zeros_like(q)))
        return (data.qfrc_bias - data.qfrc_gravcomp)[arm_qvel_ids]

    def _controller_guard(q, qd, rollout_params):
        if not controller_guard_enabled:
            return jnp.zeros_like(q)
        tau = jnp.zeros_like(q)
        joint_range = rollout_params.joint_range
        if joint_range is not None:
            q_min = joint_range[:dof, 0]
            q_max = joint_range[:dof, 1]
            err_bound = jnp.where(
                q < q_min,
                q_min - q,
                jnp.where(q > q_max, q_max - q, 0.0),
            )
            tau = tau + (joint_kp * 20.0) * err_bound
        high_vel_mask = jnp.abs(qd) > controller_guard_velocity_threshold
        tau = tau - high_vel_mask * jnp.sign(qd) * (joint_kd * 10.0) * (
            jnp.abs(qd) - controller_guard_velocity_threshold
        )
        return tau

    def _joint_impedance(q, qd, q_ref, qd_ref, rollout_params):
        return (
            joint_kp * (q_ref - q)
            + joint_kd * (qd_ref - qd)
            + _controller_guard(q, qd, rollout_params)
            + _gravity_torque(q)
        )

    def _site_pose(model, data):
        data_fwd = mjx.forward(model, data)
        rot = jnp.asarray(data_fwd.site_xmat[site_id], dtype=jnp.float32).reshape(3, 3)
        quat = matrix_to_quat_wxyz(rot)
        return data_fwd.site_xpos[site_id], quat, rot, data_fwd

    def _cart_error_for_model(model, q, target_pos, target_quat):
        data = _data_at(q, jnp.zeros_like(q))
        pos, cur_quat, cur_rot, _ = _site_pose(model, data)
        goal_quat = jnp.asarray(target_quat, dtype=jnp.float32)
        goal_quat = goal_quat / jnp.maximum(jnp.linalg.norm(goal_quat), 1e-8)
        cur_quat = cur_quat / jnp.maximum(jnp.linalg.norm(cur_quat), 1e-8)
        cur_quat = jnp.where(jnp.dot(goal_quat, cur_quat) < 0.0, -cur_quat, cur_quat)
        q_err = _quat_wxyz_mul_jax(_quat_wxyz_conj_jax(cur_quat), goal_quat)
        ori_vec_local = q_err[1:]
        ori_err = -cur_rot @ ori_vec_local
        return jnp.concatenate([pos - target_pos, ori_err], axis=0)

    def _damped_pinv(mat):
        u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
        lam = jnp.asarray(0.2, dtype=mat.dtype)
        s_inv = s / (s * s + lam * lam)
        return (vh.T * s_inv[None, :]) @ u.T

    def _osc_torque(model, data, q, qd, target_pos, target_quat, q_nullspace, rollout_params):
        pos, quat, rot, data_fwd = _site_pose(model, data)
        del quat, rot
        err = _cart_error_for_model(model, q, target_pos, target_quat)
        jac = jax.jacfwd(lambda q_in: _cart_error_for_model(model, q_in, target_pos, target_quat))(q)
        task_velocity = jac @ qd
        tau_task = jac.T @ (-osc_stiffness * err - osc_damping * task_velocity)
        jac_t_pinv = _damped_pinv(jac.T)
        nullspace_projector = jnp.eye(dof, dtype=jnp.float32) - jac.T @ jac_t_pinv
        nullspace_d = 2.0 * jnp.sqrt(jnp.maximum(osc_nullspace_stiffness, 0.0))
        tau_nullspace = nullspace_projector @ (
            osc_nullspace_stiffness * (q_nullspace - q) - nullspace_d * qd
        )
        tau = tau_task + tau_nullspace + data_fwd.qfrc_bias[arm_qvel_ids]
        tau = tau + _controller_guard(q, qd, rollout_params)
        return tau, pos

    def _step_dynamics(data, tau_cmd, rollout_params):
        model_step = rollout_params.set_mjx_model(mjx_model)
        qpos_for_act = data.qpos[actuator_dof_abs]
        qvel_for_act = data.qvel[actuator_dof_abs]
        tau_eff = actuator_util.actuator_model(
            tau_cmd,
            qpos_for_act[: tau_cmd.shape[0]],
            qvel_for_act[: tau_cmd.shape[0]],
            rollout_params.actuator_params,
        )
        ctrl = jnp.zeros((int(mjx_model.nu),), dtype=tau_eff.dtype).at[: tau_eff.shape[0]].set(tau_eff)
        return mjx.step(model_step, data.replace(ctrl=ctrl))

    def _maybe_apply_adaptor(use_adaptor, q_window, qd_window, tau_window_for_adaptor, online_state, num_updates):
        if tam_bundle is None or not use_adaptor:
            return tau_window_for_adaptor[-1], jnp.zeros_like(tau_window_for_adaptor[-1])
        tau_candidate, tau_delta = apply_online_adaptor(
            adaptor_model=adaptor_model,
            adaptor_apply_fn=adaptor_apply_fn,
            params_adaptor=model_params["adaptor"],
            q_window=q_window,
            qd_window=qd_window,
            tau_window=tau_window_for_adaptor,
            history_emb=online_state.history_emb,
            norm_stats=norm_stats,
        )
        should_use = (
            jnp.asarray(use_adaptor)
            & online_state.has_embedding
            & (num_updates >= jnp.asarray(min_updates_before_use, dtype=jnp.int32))
        )
        tau_plain = tau_window_for_adaptor[-1]
        tau_cmd = jnp.where(should_use, tau_candidate, tau_plain)
        return tau_cmd, jnp.where(should_use, tau_delta, jnp.zeros_like(tau_delta))

    def _advance_if_enabled(use_adaptor, online_state, q, qd, tau_cmd, num_updates):
        if tam_bundle is None or not use_adaptor:
            return online_state, num_updates, jnp.asarray(0, dtype=jnp.int32)
        next_state = advance_online_history_state(
            runtime,
            online_state,
            q_arm=q,
            qd_arm=qd,
            tau_arm=tau_cmd,
            raw_tau_arm=tau_cmd,
            params_hist=model_params["hist"],
            norm_stats=norm_stats,
        )
        emitted = (
            (jnp.logical_not(online_state.has_embedding) & next_state.has_embedding)
            | (next_state.next_emit_idx != online_state.next_emit_idx)
        ).astype(jnp.int32)
        next_updates = num_updates + jnp.where(use_adaptor, emitted, 0)
        next_state = jax.lax.cond(use_adaptor, lambda _: next_state, lambda _: online_state, operand=None)
        return next_state, next_updates, jnp.where(use_adaptor, emitted, 0)

    def _rollout_one(source_q, source_dq, target_pos, target_quat, rollout_params, use_adaptor: bool):
        q0 = source_q[0]
        qd0 = source_dq[0]
        data0 = _data_at(q0, qd0)
        q_window0 = jnp.repeat(q0[None, :], adaptor_seq_length, axis=0)
        qd_window0 = jnp.repeat(qd0[None, :], adaptor_seq_length, axis=0)
        tau_window0 = jnp.zeros((adaptor_seq_length, dof), dtype=jnp.float32)
        online_state0 = init_online_history_state(runtime, dtype=jnp.float32)
        num_updates0 = jnp.asarray(0, dtype=jnp.int32)

        def source_step(carry, inputs):
            data, q_window, qd_window, tau_window, online_state, num_updates = carry
            q_ref, qd_ref = inputs
            model = rollout_params.set_mjx_model(mjx_model)
            data_fwd = mjx.forward(model, data)
            q = data_fwd.qpos[arm_qpos_ids]
            qd = data_fwd.qvel[arm_qvel_ids]
            tau_plain = _joint_impedance(q, qd, q_ref, qd_ref, rollout_params)
            q_window = push_window(q_window, q)
            qd_window = push_window(qd_window, qd)
            tau_for_adaptor = push_window(tau_window, tau_plain)
            tau_cmd, tau_delta = _maybe_apply_adaptor(
                use_adaptor,
                q_window,
                qd_window,
                tau_for_adaptor,
                online_state,
                num_updates,
            )
            tau_window = tau_for_adaptor.at[-1].set(tau_cmd)
            next_online_state, next_num_updates, emitted = _advance_if_enabled(
                use_adaptor,
                online_state,
                q,
                qd,
                tau_cmd,
                num_updates,
            )
            next_data = _step_dynamics(data_fwd, tau_cmd, rollout_params)
            ee_pos, _, _, _ = _site_pose(model, data_fwd)
            return (
                next_data,
                q_window,
                qd_window,
                tau_window,
                next_online_state,
                next_num_updates,
            ), (q, qd, tau_plain, tau_cmd, tau_delta, emitted, ee_pos)

        (data_source, q_window, qd_window, tau_window, online_state, num_updates), source_log = jax.lax.scan(
            source_step,
            (data0, q_window0, qd_window0, tau_window0, online_state0, num_updates0),
            (source_q, source_dq),
        )
        q_nullspace = source_q[-1]

        def target_step(carry, inputs):
            data, q_window, qd_window, tau_window, online_state, num_updates = carry
            pos_ref, quat_ref = inputs
            model = rollout_params.set_mjx_model(mjx_model)
            data_fwd = mjx.forward(model, data)
            q = data_fwd.qpos[arm_qpos_ids]
            qd = data_fwd.qvel[arm_qvel_ids]
            tau_plain, ee_pos = _osc_torque(
                model,
                data_fwd,
                q,
                qd,
                pos_ref,
                quat_ref,
                q_nullspace,
                rollout_params,
            )
            q_window = push_window(q_window, q)
            qd_window = push_window(qd_window, qd)
            tau_for_adaptor = push_window(tau_window, tau_plain)
            tau_cmd, tau_delta = _maybe_apply_adaptor(
                use_adaptor,
                q_window,
                qd_window,
                tau_for_adaptor,
                online_state,
                num_updates,
            )
            tau_window = tau_for_adaptor.at[-1].set(tau_cmd)
            next_online_state, next_num_updates, emitted = _advance_if_enabled(
                use_adaptor,
                online_state,
                q,
                qd,
                tau_cmd,
                num_updates,
            )
            next_data = _step_dynamics(data_fwd, tau_cmd, rollout_params)
            return (
                next_data,
                q_window,
                qd_window,
                tau_window,
                next_online_state,
                next_num_updates,
            ), (q, qd, tau_plain, tau_cmd, tau_delta, emitted, ee_pos)

        (_data_target, _q_window, _qd_window, _tau_window, _online_state, final_num_updates), target_log = jax.lax.scan(
            target_step,
            (data_source, q_window, qd_window, tau_window, online_state, num_updates),
            (target_pos, target_quat),
        )
        del _data_target, _q_window, _qd_window, _tau_window, _online_state
        return {
            "source_q": source_log[0],
            "source_qd": source_log[1],
            "source_ee_pos": source_log[6],
            "target_q": target_log[0],
            "target_qd": target_log[1],
            "target_tau_plain": target_log[2],
            "target_tau_cmd": target_log[3],
            "target_tau_delta": target_log[4],
            "target_embedding_emitted": target_log[5],
            "target_ee_pos": target_log[6],
            "num_embedding_updates": final_num_updates,
        }

    vmapped_rollout = jax.jit(jax.vmap(_rollout_one, in_axes=(0, 0, 0, 0, 0, None)), static_argnums=5)
    return vmapped_rollout


def _run_batched_table(args: argparse.Namespace, conditions: Sequence[SimConditionSpec], run_dir: Path) -> Path:
    keys = {spec.key for spec in conditions}
    supported = {"direct_osc", "tam_carried"}
    if not keys.issubset(supported):
        raise SystemExit(
            "--sim-backend batched currently supports table conditions only: "
            "direct_osc and tam_carried."
        )
    import jax
    import jax.numpy as jnp

    dof = int(getattr(args, "arm_dof", 0) or len(np.asarray(args.initial_q, dtype=np.float64).reshape(-1)))
    initial_q = np.asarray(args.initial_q, dtype=np.float64).reshape(dof)
    n_iter = max(int(args.num_iterations), 1)
    iteration_offset = int(args.iteration_offset)
    refs = [
        _randomized_iteration_reference(
            args=args,
            iteration=iteration,
            initial_q=initial_q,
        )
        for iteration in range(iteration_offset, iteration_offset + n_iter)
    ]
    for ref in refs:
        print(
            f"[sim-iter {ref.iteration}] source_seed={ref.source_seed}, sim_seed={ref.sim_seed}, "
            f"waypoints={np.asarray(ref.osc_waypoint_xyz)}, "
            f"rpy_deg={np.asarray(ref.osc_waypoint_rpy_deg)}",
            flush=True,
        )

    need_tam = "tam_carried" in keys
    tam_bundle = _load_fast_tam_bundle(args) if need_tam else None
    if tam_bundle is not None:
        if tam_bundle.ablation_mode != "tam":
            raise RuntimeError(
                "The public batched source-to-OSC simulation supports only TAM "
                f"checkpoints; loaded cfg.ablation_mode={tam_bundle.ablation_mode!r}."
            )
        args.adaptor_seq_length = int(tam_bundle.adaptor_seq_length)
        print(
            f"[sim-batched] Loaded TAM checkpoint ablation_mode={tam_bundle.ablation_mode}, "
            f"adaptor_seq_length={tam_bundle.adaptor_seq_length}.",
            flush=True,
        )
    runner = _make_batched_source_to_osc_runner(args=args, tam_bundle=tam_bundle)
    requested_batch_size = int(getattr(args, "batched_eval_batch_size", 0) or 0)
    batch_size = n_iter if requested_batch_size <= 0 else max(requested_batch_size, 1)
    print(
        f"[sim-batched] Running {n_iter} trajectories on {jax.default_backend()} "
        f"in chunks of {batch_size}...",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    for chunk_start in range(0, n_iter, batch_size):
        chunk_refs = refs[chunk_start : chunk_start + batch_size]
        ideal_params, perturbed_params, chunk_setup_rows = _sample_params_for_references(
            args=args,
            refs=chunk_refs,
        )
        setup_rows.extend(chunk_setup_rows)
        (run_dir / "sim_setup.json").write_text(
            json.dumps(setup_rows, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        source_q = jnp.asarray(np.stack([ref.source_q for ref in chunk_refs], axis=0), dtype=jnp.float32)
        source_dq = jnp.asarray(np.stack([ref.source_dq for ref in chunk_refs], axis=0), dtype=jnp.float32)
        target_pos = jnp.asarray(np.stack([ref.target_pos for ref in chunk_refs], axis=0), dtype=jnp.float32)
        target_quat = jnp.asarray(np.stack([ref.target_quat for ref in chunk_refs], axis=0), dtype=jnp.float32)

        print(
            f"[sim-batched] chunk {chunk_start // batch_size + 1}: "
            f"iterations {chunk_refs[0].iteration}-{chunk_refs[-1].iteration}",
            flush=True,
        )
        ideal_out = runner(source_q, source_dq, target_pos, target_quat, ideal_params, False)
        direct_out = runner(source_q, source_dq, target_pos, target_quat, perturbed_params, False)
        tam_out = (
            runner(source_q, source_dq, target_pos, target_quat, perturbed_params, True)
            if need_tam
            else None
        )
        ideal_out = jax.tree_util.tree_map(lambda x: np.asarray(x), ideal_out)
        direct_out = jax.tree_util.tree_map(lambda x: np.asarray(x), direct_out)
        tam_out_np = None if tam_out is None else jax.tree_util.tree_map(lambda x: np.asarray(x), tam_out)
        target_pos_np = np.asarray(target_pos)
        source_q_np = np.asarray(source_q)

        condition_outputs = {"direct_osc": direct_out}
        if tam_out_np is not None:
            condition_outputs["tam_carried"] = tam_out_np
        for spec in conditions:
            out = condition_outputs[spec.key]
            metrics = _batched_target_metrics_np(
                target_ee=out["target_ee_pos"],
                target_ref=target_pos_np,
                ideal_ee=ideal_out["target_ee_pos"],
            )
            source_metrics = _batched_source_vs_ideal_metrics_np(
                source_q=out["source_q"],
                source_ee=out["source_ee_pos"],
                ideal_source_q=ideal_out["source_q"],
                ideal_source_ee=ideal_out["source_ee_pos"],
            )
            for idx, ref in enumerate(chunk_refs):
                num_updates = int(np.asarray(out["num_embedding_updates"])[idx]) if spec.key == "tam_carried" else 0
                row = {
                    "iteration": int(ref.iteration),
                    "source_seed": int(ref.source_seed),
                    "sim_seed": int(ref.sim_seed),
                    "osc_delta_xyz": list(ref.osc_delta_xyz),
                    "osc_delta_rpy_deg": list(ref.osc_delta_rpy_deg),
                    "osc_waypoint_xyz": [list(w) for w in ref.osc_waypoint_xyz],
                    "osc_waypoint_rpy_deg": [list(w) for w in ref.osc_waypoint_rpy_deg],
                    "condition": spec.label,
                    "condition_key": spec.key,
                    "source_segment": spec.source_segment,
                    "switch_behavior": spec.switch_behavior,
                    "target_osc_segment": spec.target_segment,
                    "target_ee_samples": int(out["target_ee_pos"].shape[1]),
                    "target_vs_reference_ee_pos_samples": int(out["target_ee_pos"].shape[1]),
                    "target_ideal_ee_samples": int(ideal_out["target_ee_pos"].shape[1]),
                    "target_vs_ideal_ee_pos_samples": int(out["target_ee_pos"].shape[1]),
                    "target_ee_pos_rmse_m": float(metrics["target_ee_pos_rmse_m"][idx]),
                    "target_ee_final_error_m": float(metrics["target_ee_final_error_m"][idx]),
                    "target_vs_reference_ee_pos_rmse_m": float(metrics["target_ee_pos_rmse_m"][idx]),
                    "target_vs_reference_ee_final_error_m": float(metrics["target_ee_final_error_m"][idx]),
                    "target_vs_ideal_ee_pos_rmse_m": float(metrics["target_vs_ideal_ee_pos_rmse_m"][idx]),
                    "target_vs_ideal_ee_final_error_m": float(metrics["target_vs_ideal_ee_final_error_m"][idx]),
                    "target_vs_reference_ee_ori_samples": 0,
                    "target_vs_reference_ee_ori_rmse_deg": float("nan"),
                    "target_vs_reference_ee_final_ori_error_deg": float("nan"),
                    "target_vs_ideal_ee_ori_samples": 0,
                    "target_vs_ideal_ee_ori_rmse_deg": float("nan"),
                    "target_vs_ideal_ee_final_ori_error_deg": float("nan"),
                    "source_ideal_joint_samples": int(source_metrics["source_ideal_joint_samples"][idx]),
                    "source_vs_ideal_joint_pos_samples": int(
                        source_metrics["source_vs_ideal_joint_pos_samples"][idx]
                    ),
                    "source_vs_ideal_joint_pos_rmse_rad": float(
                        source_metrics["source_vs_ideal_joint_pos_rmse_rad"][idx]
                    ),
                    "source_vs_ideal_joint_pos_rmse_deg": float(
                        source_metrics["source_vs_ideal_joint_pos_rmse_deg"][idx]
                    ),
                    "source_vs_ideal_joint_final_error_rad": float(
                        source_metrics["source_vs_ideal_joint_final_error_rad"][idx]
                    ),
                    "source_ideal_ee_samples": int(source_metrics["source_ideal_ee_samples"][idx]),
                    "source_vs_ideal_ee_pos_samples": int(
                        source_metrics["source_vs_ideal_ee_pos_samples"][idx]
                    ),
                    "source_vs_ideal_ee_pos_rmse_m": float(
                        source_metrics["source_vs_ideal_ee_pos_rmse_m"][idx]
                    ),
                    "source_vs_ideal_ee_final_error_m": float(
                        source_metrics["source_vs_ideal_ee_final_error_m"][idx]
                    ),
                    "source_final_q_error_rad": float(
                        np.linalg.norm(out["source_q"][idx, -1] - source_q_np[idx, -1])
                    ),
                    "switch_actual_to_osc_start_pos_error_m": float(
                        np.linalg.norm(out["source_ee_pos"][idx, -1] - target_pos_np[idx, 0])
                    ),
                    "controller_side_guard": bool(args.controller_side_guard),
                    "controller_guard_velocity_threshold": float(args.controller_guard_velocity_threshold),
                    "tam_embeddings_total": num_updates,
                    "tam_embeddings_sent": num_updates,
                }
                rows.append(row)
                print(
                    f"[sim-batched iter {ref.iteration}][{spec.label}] "
                    f"cmd_rmse={row['target_ee_pos_rmse_m']:.6f}m, "
                    f"ideal_rmse={row['target_vs_ideal_ee_pos_rmse_m']:.6f}m, "
                    f"updates={num_updates}",
                    flush=True,
                )
        _write_summary_files(run_dir, rows)
        _write_aggregate_files(run_dir, rows)

    _write_summary_files(run_dir, rows)
    _write_aggregate_files(run_dir, rows)
    print(f"[done] Batched sim logs and table saved to {run_dir}")
    return run_dir


def _rollout_param_summary(params: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            summary[field.name] = {"shape": list(arr.shape)}
            continue
        arr_f = arr.astype(np.float64, copy=False)
        finite = arr_f[np.isfinite(arr_f)]
        if finite.size == 0:
            summary[field.name] = {"shape": list(arr.shape)}
            continue
        summary[field.name] = {
            "shape": list(arr.shape),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "l2": float(np.linalg.norm(finite)),
        }
    return summary


def _install_optional_viser_stub() -> None:
    try:
        __import__("viser")
    except ModuleNotFoundError:
        stub = types.ModuleType("viser")

        class _UnavailableViserServer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("viser is not installed in this environment.")

        stub.ViserServer = _UnavailableViserServer  # type: ignore[attr-defined]
        sys.modules["viser"] = stub
    try:
        __import__("yourdfpy")
    except ModuleNotFoundError:
        yourdfpy_stub = types.ModuleType("yourdfpy")
        urdf_stub = types.ModuleType("yourdfpy.urdf")
        yourdfpy_stub.urdf = urdf_stub  # type: ignore[attr-defined]
        sys.modules["yourdfpy"] = yourdfpy_stub
        sys.modules["yourdfpy.urdf"] = urdf_stub


def _build_tam_state(args: argparse.Namespace) -> OnlineTamSimState:
    _install_optional_viser_stub()
    from simadaptor.deploy.history_runtime import RealTimeHistoryAdaptor

    ckpt_args = [
        args.adaptor_ckpt_path is not None,
    ]
    if sum(1 for x in ckpt_args if x) != 1:
        raise SystemExit(
            "Simulation TAM rows need --tam-ckpt-path."
        )

    runtime = RealTimeHistoryAdaptor(
        simadaptor_ckpt_path=str(args.adaptor_ckpt_path) if args.adaptor_ckpt_path is not None else None,
        xml_path=args.xml,
        expected_dt=float(args.dt),
    )
    resolved_history_mode = _resolve_history_torque_mode(runtime)
    args.resolved_history_torque_mode = resolved_history_mode
    sim_inf = runtime.inf
    if sim_inf is None:
        raise RuntimeError("RealTimeHistoryAdaptor did not expose SimAdaptorInference.")
    cfg = getattr(sim_inf, "cfg", None)
    ablation_mode = str(getattr(cfg, "ablation_mode", "tam") or "tam")
    dagger_cfg = getattr(sim_inf, "dagger_cfg", None)
    # DAgger-finetuned checkpoints are TAM-compatible even when their config
    # predates the 'tam' mode name.
    if ablation_mode not in {"tam", "full_mam"} and dagger_cfg is None:
        raise RuntimeError(
            "The public source-to-OSC simulation supports only TAM checkpoints; "
            f"loaded cfg.ablation_mode={ablation_mode!r}."
        )
    _history_apply_fn, adaptor_apply_jit, model_params, norm_stats, _cfg = sim_inf.get_apply_fns()
    del _history_apply_fn, _cfg
    base_runtime = None
    tam_runtime = None
    history_fusion_params = None
    if resolved_history_mode == "base_tam_fusion":
        history_fusion_params = _history_fusion_params_from_inf(sim_inf)
        stream_kwargs = dict(
            sim_inf=sim_inf,
            runtime_bundle=runtime.runtime_bundle,
            expected_dt=float(args.dt),
        )
        base_runtime = RealTimeHistoryAdaptor(**stream_kwargs)
        tam_runtime = RealTimeHistoryAdaptor(**stream_kwargs)
        print(
            "[sim] base_tam_fusion history: streaming applied/base/TAM torque "
            "histories through weight-sharing encoders with a linear fusion layer.",
            flush=True,
        )
    state = OnlineTamSimState(
        runtime=runtime,
        adaptor_apply_jit=adaptor_apply_jit,
        model_params=model_params,
        norm_stats=norm_stats,
        min_patches_before_send=int(args.min_patches_before_send),
        embedding_interval_s=float(args.embedding_interval_s),
        enable_after_first_embedding=bool(args.enable_after_first_embedding),
        history_torque_mode=resolved_history_mode,
        base_runtime=base_runtime,
        tam_runtime=tam_runtime,
        history_fusion_params=history_fusion_params,
    )
    state.reset()
    return state


def _randomized_iteration_reference(
    *,
    args: argparse.Namespace,
    iteration: int,
    initial_q: np.ndarray,
) -> SimReference:
    randomize = bool(args.randomize_source_target or int(args.num_iterations) > 1)
    dof = int(initial_q.reshape(-1).shape[0])
    source_seed = int(args.source_seed) + int(iteration) * 17
    sim_seed = int(args.sim_seed) + int(iteration) * 1009
    rng = np.random.default_rng(int(args.reference_seed) + int(iteration))

    source_amp_deg_raw = getattr(args, "source_amp_deg", None)
    if source_amp_deg_raw is None:
        source_amp_deg_raw = (30.0, 25.0, 30.0, 20.0, 30.0, 25.0, 34.0)
    amp_deg = np.asarray(source_amp_deg_raw, dtype=np.float64).reshape(-1)
    amp_deg = np.asarray(_resize_float_vector(amp_deg, dof, name="source_amp_deg"), dtype=np.float64)
    source_cycles_raw = getattr(args, "source_cycles", None)
    if source_cycles_raw is None:
        source_cycles_raw = (3, 3, 3, 3, 3, 4, 6)
    cycle_vals = np.asarray(source_cycles_raw, dtype=np.int64).reshape(-1)
    cycle_vals = np.asarray(_resize_int_vector(cycle_vals, dof, name="source_cycles"), dtype=np.int64)
    if randomize:
        amp_scale = rng.uniform(
            float(args.source_amp_scale_min),
            float(args.source_amp_scale_max),
            size=dof,
        )
        amp_deg = amp_deg * amp_scale
        cycle_vals = rng.integers(
            int(args.source_cycle_min),
            int(args.source_cycle_max) + 1,
            size=dof,
        )

    waypoint_xyz_arg = getattr(args, "osc_waypoint_xyz", None)
    waypoint_rpy_arg = getattr(args, "osc_waypoint_rpy_deg", None)
    if (waypoint_xyz_arg is None) != (waypoint_rpy_arg is None):
        raise ValueError("--osc-waypoint-xyz and --osc-waypoint-rpy-deg must be provided together")
    if waypoint_xyz_arg is not None:
        waypoint_xyz = np.asarray(waypoint_xyz_arg, dtype=np.float64).reshape(-1, 3)
        waypoint_rpy_deg = np.asarray(waypoint_rpy_arg, dtype=np.float64).reshape(-1, 3)
        if waypoint_xyz.shape[0] != waypoint_rpy_deg.shape[0]:
            raise ValueError("--osc-waypoint-xyz and --osc-waypoint-rpy-deg must have the same count")
    elif bool(args.osc_sample_waypoints):
        waypoint_xyz, waypoint_rpy_deg = sample_osc_waypoints(
            rng=rng,
            num_waypoints=int(args.osc_num_waypoints),
            xyz_min=args.osc_waypoint_xyz_min,
            xyz_max=args.osc_waypoint_xyz_max,
            rpy_deg_min=args.osc_waypoint_rpy_deg_min,
            rpy_deg_max=args.osc_waypoint_rpy_deg_max,
        )
    else:
        if randomize:
            delta_min = np.asarray(args.osc_delta_xyz_min, dtype=np.float64).reshape(3)
            delta_max = np.asarray(args.osc_delta_xyz_max, dtype=np.float64).reshape(3)
            lo = np.minimum(delta_min, delta_max)
            hi = np.maximum(delta_min, delta_max)
            delta = rng.uniform(lo, hi)
            min_norm = float(args.osc_delta_min_norm)
            norm = float(np.linalg.norm(delta))
            if norm < min_norm:
                direction = rng.normal(size=3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm < 1e-9:
                    direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    direction = direction / direction_norm
                delta = direction * min_norm
                delta = np.clip(delta, lo, hi)
        else:
            delta = np.asarray(args.osc_delta_xyz, dtype=np.float64).reshape(3)
        waypoint_xyz = np.asarray([delta], dtype=np.float64)
        waypoint_rpy_deg = np.asarray([args.osc_delta_rpy_deg], dtype=np.float64).reshape(1, 3)
    target_delta_xyz = np.asarray(waypoint_xyz[-1], dtype=np.float64).reshape(3)
    target_delta_rpy_deg = np.asarray(waypoint_rpy_deg[-1], dtype=np.float64).reshape(3)

    source_t, source_q, source_dq = make_source_reference(
        start_q=initial_q,
        duration_s=float(args.source_duration_s),
        dt_s=float(args.dt),
        amp_deg=tuple(float(x) for x in amp_deg),
        cycles=tuple(int(x) for x in cycle_vals),
        seed=source_seed,
    )

    osc_start_pos, osc_start_quat = fk_site_pose(
        xml_path=args.xml.expanduser().resolve(),
        q=source_q[-1],
        site_name=str(args.fk_site),
    )
    target_t, target_pos, target_quat = make_cartesian_target_reference(
        start_pos=osc_start_pos,
        start_quat_wxyz=osc_start_quat,
        duration_s=float(args.target_duration_s),
        dt_s=float(args.dt),
        delta_xyz=tuple(float(x) for x in target_delta_xyz),
        delta_rpy_deg=tuple(float(x) for x in target_delta_rpy_deg),
        waypoint_xyz=waypoint_xyz,
        waypoint_rpy_deg=waypoint_rpy_deg,
    )
    return SimReference(
        iteration=int(iteration),
        source_seed=int(source_seed),
        sim_seed=int(sim_seed),
        source_amp_deg=tuple(float(x) for x in amp_deg),
        source_cycles=tuple(int(x) for x in cycle_vals),
        osc_delta_xyz=tuple(float(x) for x in waypoint_xyz[-1]),
        osc_delta_rpy_deg=tuple(float(x) for x in waypoint_rpy_deg[-1]),
        osc_waypoint_xyz=tuple(tuple(float(v) for v in row) for row in waypoint_xyz),
        osc_waypoint_rpy_deg=tuple(tuple(float(v) for v in row) for row in waypoint_rpy_deg),
        source_t=source_t,
        source_q=source_q,
        source_dq=source_dq,
        target_t=target_t,
        target_pos=target_pos,
        target_quat=target_quat,
        osc_start_pos=osc_start_pos,
        osc_start_quat=osc_start_quat,
    )


def _make_models(
    *,
    xml_path: Path,
    dt_s: float,
    rollout_params: Any,
) -> tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjModel, mujoco.MjData]:
    controller_model = mujoco.MjModel.from_xml_path(str(xml_path))
    controller_model.opt.timestep = float(dt_s)
    controller_model.body_gravcomp[:] = 0.0
    _remove_mujoco_robot_limits(controller_model)
    controller_data = mujoco.MjData(controller_model)

    plant_model = mujoco.MjModel.from_xml_path(str(xml_path))
    plant_model.opt.timestep = float(dt_s)
    plant_model.body_gravcomp[:] = 0.0
    _remove_mujoco_robot_limits(plant_model)
    _apply_rollout_params_to_model(plant_model, rollout_params)
    plant_data = mujoco.MjData(plant_model)
    return plant_model, plant_data, controller_model, controller_data


def _simulate_condition(
    *,
    args: argparse.Namespace,
    spec: Any,
    run_dir: Path,
    condition_dir_name: Optional[str] = None,
    iteration_meta: Optional[dict[str, Any]] = None,
    ideal_source_log_np: Optional[dict[str, np.ndarray]] = None,
    ideal_target_log_np: Optional[dict[str, np.ndarray]] = None,
    source_t: np.ndarray,
    source_q: np.ndarray,
    source_dq: np.ndarray,
    target_t: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    initial_q: np.ndarray,
    joint_kp: np.ndarray,
    joint_kd: np.ndarray,
    osc_stiffness: np.ndarray,
    osc_damping: np.ndarray,
    osc_nullspace_stiffness: float,
    rollout_params: Any,
    tam_state: Optional[OnlineTamSimState],
) -> dict[str, Any]:
    plant_model, plant_data, controller_model, controller_data = _make_models(
        xml_path=args.xml.expanduser().resolve(),
        dt_s=float(args.dt),
        rollout_params=rollout_params,
    )
    dof = int(np.asarray(initial_q, dtype=np.float64).reshape(-1).shape[0])
    qpos_idx, qvel_idx, _ = _arm_indices(plant_model, dof=dof)
    ctrl_qpos_idx, ctrl_qvel_idx, _ = _arm_indices(controller_model, dof=dof)
    site_id = int(mujoco.mj_name2id(plant_model, mujoco.mjtObj.mjOBJ_SITE, str(args.fk_site)))
    if site_id < 0:
        raise ValueError(f"Site not found in XML: {args.fk_site}")
    _set_arm_state(plant_model, plant_data, qpos_idx, qvel_idx, initial_q, np.zeros(dof))
    controller_joint_range = _as_np_field(getattr(rollout_params, "joint_range", None))
    controller_guard_enabled = bool(getattr(args, "controller_side_guard", True))
    controller_guard_velocity_threshold = float(
        getattr(args, "controller_guard_velocity_threshold", 4.0)
    )

    window_len = max(int(args.adaptor_seq_length), 1)
    q_short = np.repeat(initial_q.astype(np.float32)[None, :], window_len, axis=0)
    dq_short = np.zeros_like(q_short)
    tau_short = np.zeros_like(q_short)
    chunk_t: list[float] = []
    chunk_q: list[np.ndarray] = []
    chunk_dq: list[np.ndarray] = []
    chunk_tau: list[np.ndarray] = []
    chunk_tau_base: list[np.ndarray] = []
    chunk_tau_delta: list[np.ndarray] = []

    def clear_chunk() -> None:
        chunk_t.clear()
        chunk_q.clear()
        chunk_dq.clear()
        chunk_tau.clear()
        chunk_tau_base.clear()
        chunk_tau_delta.clear()

    def flush_chunk(active_tam: Optional[OnlineTamSimState]) -> None:
        if active_tam is None:
            clear_chunk()
            return
        _maybe_update_tam(
            active_tam,
            timestamps=chunk_t,
            q_rows=chunk_q,
            dq_rows=chunk_dq,
            tau_rows=chunk_tau,
            tau_base_rows=chunk_tau_base,
            tau_delta_rows=chunk_tau_delta,
        )
        clear_chunk()

    uses_tam = _condition_uses_tam(spec)

    if spec.tam_on_source:
        assert tam_state is not None
        tam_state.reset()
        active_source_tam = tam_state
    else:
        active_source_tam = None

    source_log = _empty_log()
    for step_idx, t_now in enumerate(source_t):
        q = np.asarray(plant_data.qpos[qpos_idx], dtype=np.float64)
        dq = np.asarray(plant_data.qvel[qvel_idx], dtype=np.float64)
        mujoco.mj_forward(plant_model, plant_data)
        tau_plain = _joint_impedance_torque(
            controller_model=controller_model,
            controller_data=controller_data,
            qpos_idx=ctrl_qpos_idx,
            qvel_idx=ctrl_qvel_idx,
            q=q,
            dq=dq,
            q_ref=source_q[step_idx],
            dq_ref=source_dq[step_idx],
            kp=joint_kp,
            kd=joint_kd,
            joint_range=controller_joint_range,
            controller_guard_enabled=controller_guard_enabled,
            velocity_threshold=controller_guard_velocity_threshold,
        )
        q_short = _push_short(q_short, q.astype(np.float32))
        dq_short = _push_short(dq_short, dq.astype(np.float32))
        tau_plain_window = _push_short(tau_short, tau_plain.astype(np.float32))
        tau_cmd, tau_delta = _apply_tam_if_available(
            active_source_tam,
            q_window=q_short,
            dq_window=dq_short,
            tau_plain_window=tau_plain_window,
        )
        tau_short = tau_plain_window.copy()
        tau_short[-1] = np.asarray(tau_cmd, dtype=np.float32)

        chunk_t.append(float(t_now))
        chunk_q.append(q.astype(np.float32))
        chunk_dq.append(dq.astype(np.float32))
        chunk_tau.append(np.asarray(tau_cmd, dtype=np.float32))
        chunk_tau_base.append(
            np.asarray(tau_cmd, dtype=np.float32) - np.asarray(tau_delta, dtype=np.float32)
        )
        chunk_tau_delta.append(np.asarray(tau_delta, dtype=np.float32))
        if len(chunk_t) >= int(args.window_rows):
            flush_chunk(active_source_tam)

        tau_eff = _actuator_model_np(tau_cmd, q, dq, rollout_params)
        plant_data.ctrl[:dof] = _clip_to_actuator_range(plant_model, tau_eff)
        ee_pos = np.asarray(plant_data.site_xpos[site_id], dtype=np.float64).copy()
        ee_quat = _site_quat_wxyz(plant_data, site_id)
        _append_log(
            source_log,
            t=float(t_now),
            q=q,
            dq=dq,
            target_q=source_q[step_idx],
            target_dq=source_dq[step_idx],
            tau_plain=tau_plain,
            tau_cmd=tau_cmd,
            tau_delta=tau_delta,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            tam_enabled=bool(active_source_tam is not None and active_source_tam.enabled),
        )
        if step_idx < len(source_t) - 1:
            mujoco.mj_step(plant_model, plant_data)

    flush_chunk(active_source_tam)

    target_log = _empty_log()
    if not bool(getattr(args, "source_only", False)):
        if bool(spec.reset_tam_at_switch):
            if uses_tam:
                assert tam_state is not None
                tam_state.reset()
            q_now = np.asarray(plant_data.qpos[qpos_idx], dtype=np.float64)
            dq_now = np.asarray(plant_data.qvel[qvel_idx], dtype=np.float64)
            q_short = np.repeat(q_now.astype(np.float32)[None, :], window_len, axis=0)
            dq_short = np.repeat(dq_now.astype(np.float32)[None, :], window_len, axis=0)
            tau_short = np.zeros_like(q_short)
        elif spec.tam_on_target:
            assert tam_state is not None
        else:
            tam_state = None

        active_target_tam = tam_state if spec.tam_on_target else None
        target_t_offset = float(source_t[-1])
        q_nullspace = np.asarray(source_q[-1], dtype=np.float64).reshape(dof)
        for step_idx, t_rel in enumerate(target_t):
            t_now = target_t_offset + float(t_rel)
            q = np.asarray(plant_data.qpos[qpos_idx], dtype=np.float64)
            dq = np.asarray(plant_data.qvel[qvel_idx], dtype=np.float64)
            mujoco.mj_forward(plant_model, plant_data)
            tau_plain, ee_pos, ee_quat = _osc_torque(
                plant_model=plant_model,
                plant_data=plant_data,
                qpos_idx=qpos_idx,
                qvel_idx=qvel_idx,
                site_id=site_id,
                target_pos=target_pos[step_idx],
                target_quat=target_quat[step_idx],
                stiffness=osc_stiffness,
                damping=osc_damping,
                q_nullspace=q_nullspace,
                nullspace_stiffness=osc_nullspace_stiffness,
                joint_kp=joint_kp,
                joint_kd=joint_kd,
                joint_range=controller_joint_range,
                controller_guard_enabled=controller_guard_enabled,
                velocity_threshold=controller_guard_velocity_threshold,
            )
            q_short = _push_short(q_short, q.astype(np.float32))
            dq_short = _push_short(dq_short, dq.astype(np.float32))
            tau_plain_window = _push_short(tau_short, tau_plain.astype(np.float32))
            tau_cmd, tau_delta = _apply_tam_if_available(
                active_target_tam,
                q_window=q_short,
                dq_window=dq_short,
                tau_plain_window=tau_plain_window,
            )
            tau_short = tau_plain_window.copy()
            tau_short[-1] = np.asarray(tau_cmd, dtype=np.float32)

            chunk_t.append(float(t_now))
            chunk_q.append(q.astype(np.float32))
            chunk_dq.append(dq.astype(np.float32))
            chunk_tau.append(np.asarray(tau_cmd, dtype=np.float32))
            chunk_tau_base.append(
                np.asarray(tau_cmd, dtype=np.float32) - np.asarray(tau_delta, dtype=np.float32)
            )
            chunk_tau_delta.append(np.asarray(tau_delta, dtype=np.float32))
            if len(chunk_t) >= int(args.window_rows):
                flush_chunk(active_target_tam)

            tau_eff = _actuator_model_np(tau_cmd, q, dq, rollout_params)
            plant_data.ctrl[:dof] = _clip_to_actuator_range(plant_model, tau_eff)
            _append_log(
                target_log,
                t=float(t_rel),
                q=q,
                dq=dq,
                target_pos=target_pos[step_idx],
                target_quat=target_quat[step_idx],
                tau_plain=tau_plain,
                tau_cmd=tau_cmd,
                tau_delta=tau_delta,
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                tam_enabled=bool(active_target_tam is not None and active_target_tam.enabled),
            )
            if step_idx < len(target_t) - 1:
                mujoco.mj_step(plant_model, plant_data)

        flush_chunk(active_target_tam)

    condition_dir = run_dir / (str(condition_dir_name) if condition_dir_name is not None else spec.key)
    condition_dir.mkdir(parents=True, exist_ok=True)
    source_np = _log_to_np(source_log, dof=dof)
    target_np = _log_to_np(target_log, dof=dof)
    np.savez(condition_dir / "source_log.npz", **source_np, source_t=source_t, source_q=source_q, source_dq=source_dq)
    np.savez(
        condition_dir / "target_log.npz",
        **target_np,
        target_t=target_t,
        target_pos_ref=target_pos,
        target_quat_ref=target_quat,
        osc_nullspace_stiffness=np.asarray(float(osc_nullspace_stiffness), dtype=np.float64),
    )

    if bool(getattr(args, "source_only", False)):
        metrics = {
            "target_ee_samples": 0,
            "target_ee_pos_rmse_m": float("nan"),
            "target_ee_final_error_m": float("nan"),
            "target_ideal_ee_samples": 0,
            "target_vs_ideal_ee_pos_rmse_m": float("nan"),
            "target_vs_ideal_ee_final_error_m": float("nan"),
            "target_vs_ideal_ee_ori_rmse_deg": float("nan"),
            "target_vs_ideal_ee_final_ori_error_deg": float("nan"),
        }
    else:
        metrics = _target_metrics_from_np(
            target_log_np=target_np,
            target_t=target_t,
            target_pos=target_pos,
        )
    if ideal_target_log_np is not None and not bool(getattr(args, "source_only", False)):
        metrics.update(
            _trajectory_error_metrics(
                target_log_np=target_np,
                ideal_log_np=ideal_target_log_np,
            )
        )
    else:
        metrics.update(
            {
                "target_ideal_ee_samples": 0,
                "target_vs_ideal_ee_pos_rmse_m": float("nan"),
                "target_vs_ideal_ee_final_error_m": float("nan"),
            }
        )
    metrics.update(
        _source_vs_ideal_metrics(
            source_log_np=source_np,
            ideal_source_log_np=ideal_source_log_np,
        )
    )
    source_final_q_error = float(np.linalg.norm(np.asarray(source_np["q"][-1], dtype=np.float64) - source_q[-1]))
    switch_pos_error = float(np.linalg.norm(np.asarray(source_np["ee_pos"][-1], dtype=np.float64) - target_pos[0]))
    report_tam_state = tam_state if uses_tam else None
    row = {
        **(iteration_meta or {}),
        "condition": spec.label,
        "condition_key": spec.key,
        "source_segment": spec.source_segment,
        "switch_behavior": spec.switch_behavior,
        "target_osc_segment": spec.target_segment,
        "source_final_q_error_rad": source_final_q_error,
        "switch_actual_to_osc_start_pos_error_m": switch_pos_error,
        "tam_embeddings_total": int(report_tam_state.num_embeddings) if report_tam_state is not None else 0,
        "tam_embeddings_sent": int(report_tam_state.num_sent) if report_tam_state is not None else 0,
        "controller_side_guard": bool(controller_guard_enabled),
        "controller_guard_velocity_threshold": float(controller_guard_velocity_threshold),
        **metrics,
    }
    (condition_dir / "metrics.json").write_text(
        json.dumps(row, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulated source-history to OSC switch experiment for direct OSC and TAM.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
        help=(
            "Subset: all, tam_all, direct_osc, tam_reset, tam_carried."
        ),
    )
    parser.add_argument(
        "--sim-backend",
        choices=("auto", "legacy", "batched"),
        default="auto",
        help=(
            "Simulation backend. 'batched' runs direct_osc/tam_carried table rows "
            "inside one JAX/MJX vectorized loop; 'legacy' keeps per-condition Python logs."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=Path("eval_logs") / "source_to_osc_tam_sim")
    parser.add_argument(
        "--robot-preset",
        choices=(
            "auto",
            "panda",
            "piper",
            "rby1",
            "rby1_onearm",
            "kuka",
            "kuka_iiwa14",
            "iiwa14",
            "google",
            "google_robot",
            "unitree",
            "unitree_z1",
            "z1",
            "flexiv",
            "flexiv_rizon4",
            "rizon4",
        ),
        default="auto",
        help="Convenience preset for robot XML/profile/default joint vectors.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_PANDA_XML)
    parser.add_argument("--fk-site", type=str, default="gripper")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation/control timestep.")
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of randomized simulation iterations.")
    parser.add_argument(
        "--batched-eval-batch-size",
        type=int,
        default=0,
        help="Chunk size for --sim-backend batched. 0 runs all iterations in one vectorized batch.",
    )
    parser.add_argument(
        "--iteration-offset",
        type=int,
        default=0,
        help="Global iteration index offset for sharded runs across machines.",
    )
    parser.add_argument("--window-rows", type=int, default=50, help="Rows per online TAM history push.")
    parser.add_argument(
        "--tam-seq-length",
        dest="adaptor_seq_length",
        type=int,
        default=0,
        help="Short torque window length for TAM input. 0 infers it from the checkpoint.",
    )

    parser.add_argument(
        "--initial-q",
        type=parse_optional_float_vec,
        default=None,
        help="Comma-separated initial arm joint configuration. Defaults from the selected robot.",
    )
    parser.add_argument("--source-duration-s", type=float, default=16.0)
    parser.add_argument(
        "--source-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only the source joint trajectory and source-vs-ideal metrics, skipping the target OSC segment.",
    )
    parser.add_argument("--source-amp-deg", type=parse_float_vec, default=None)
    parser.add_argument("--source-cycles", type=int, nargs="+", default=None)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--reference-seed", type=int, default=12345)
    parser.add_argument("--randomize-source-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-amp-scale-min", type=float, default=0.75)
    parser.add_argument("--source-amp-scale-max", type=float, default=1.25)
    parser.add_argument("--source-cycle-min", type=int, default=3)
    parser.add_argument("--source-cycle-max", type=int, default=7)
    parser.add_argument(
        "--param-debug-mode",
        choices=("default", "actuator_major_no_payload"),
        default="default",
        help=(
            "Optional source-to-OSC diagnostic parameter override. "
            "'actuator_major_no_payload' preserves the sampled actuator/controller "
            "perturbations but resets perturbed body mass, inertia, and ipos to the "
            "ideal model."
        ),
    )
    parser.add_argument("--joint-stiffness", type=parse_float_vec, default=None)
    parser.add_argument("--joint-damping", type=parse_optional_float_vec, default=None)
    parser.add_argument(
        "--controller-side-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the datagen controller-side safety guard to commanded torques: "
            "one-sided soft joint-limit pullback from rollout_params.joint_range plus "
            "high-velocity damping. XML-side limits remain disabled."
        ),
    )
    parser.add_argument(
        "--controller-guard-velocity-threshold",
        type=float,
        default=4.0,
        help="Velocity threshold in rad/s for the controller-side safety damping guard.",
    )

    parser.add_argument("--target-duration-s", type=float, default=8.0)
    parser.add_argument("--osc-delta-xyz", type=parse_vec3, default=DEFAULT_SIM_OSC_WAYPOINT_XYZ[-1])
    parser.add_argument(
        "--osc-delta-rpy-deg",
        type=parse_vec3,
        default=DEFAULT_SIM_OSC_WAYPOINT_RPY_DEG[-1],
        help="End-of-trajectory roll,pitch,yaw offset in degrees, applied in the start end-effector frame.",
    )
    parser.add_argument(
        "--osc-waypoint-xyz",
        type=parse_vec3,
        nargs="+",
        default=None,
        help="Optional explicit OSC waypoint position offsets from the start pose, in meters.",
    )
    parser.add_argument(
        "--osc-waypoint-rpy-deg",
        type=parse_vec3,
        nargs="+",
        default=None,
        help="Optional explicit OSC waypoint roll,pitch,yaw offsets from the start orientation, in degrees.",
    )
    parser.add_argument("--osc-num-waypoints", type=int, default=5)
    parser.add_argument("--osc-sample-waypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--osc-waypoint-xyz-min", type=parse_vec3, default=DEFAULT_OSC_WAYPOINT_XYZ_MIN)
    parser.add_argument("--osc-waypoint-xyz-max", type=parse_vec3, default=DEFAULT_OSC_WAYPOINT_XYZ_MAX)
    parser.add_argument("--osc-waypoint-rpy-deg-min", type=parse_vec3, default=DEFAULT_OSC_WAYPOINT_RPY_DEG_MIN)
    parser.add_argument("--osc-waypoint-rpy-deg-max", type=parse_vec3, default=DEFAULT_OSC_WAYPOINT_RPY_DEG_MAX)
    parser.add_argument("--osc-delta-xyz-min", type=parse_vec3, default=(-0.05, -0.035, -0.02))
    parser.add_argument("--osc-delta-xyz-max", type=parse_vec3, default=(0.05, 0.035, 0.02))
    parser.add_argument("--osc-delta-min-norm", type=float, default=0.02)
    parser.add_argument("--profile-table", type=Path, default=DEFAULT_PROFILE_TABLE)
    parser.add_argument("--profile-key", type=str, default=None)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--osc-stiffness", type=parse_vec6, default=(400.0, 400.0, 400.0, 80.0, 80.0, 80.0))
    parser.add_argument("--osc-damping", type=parse_vec6, default=(40.0, 40.0, 40.0, 10.0, 10.0, 10.0))
    parser.add_argument(
        "--osc-nullspace-stiffness",
        type=float,
        default=DEFAULT_OSC_NULLSPACE_STIFFNESS,
        help="OSC nullspace stiffness used by all sim/ideal target rollouts.",
    )
    parser.add_argument("--tam-ckpt-path", dest="adaptor_ckpt_path", type=Path, default=None)
    parser.add_argument("--min-patches-before-send", type=int, default=1)
    parser.add_argument("--embedding-interval-s", type=float, default=0.05)
    parser.add_argument("--enable-after-first-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def run(args: argparse.Namespace) -> Path:
    conditions = resolve_sim_conditions(args.conditions)
    dof = _resolve_robot_args(args)
    sim_backend = _sim_backend_choice(args, conditions)
    setattr(args, "sim_backend_resolved", sim_backend)
    run_dir = args.outdir.expanduser().resolve() / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_q = np.asarray(args.initial_q, dtype=np.float64).reshape(dof)
    joint_kp = np.asarray(args.joint_stiffness, dtype=np.float64).reshape(dof)
    if args.joint_damping is None:
        joint_kd = 2.0 * np.sqrt(joint_kp) * _default_damping_ratio(dof)
    else:
        joint_kd = np.asarray(args.joint_damping, dtype=np.float64).reshape(dof)
    osc_stiffness = np.asarray(args.osc_stiffness, dtype=np.float64).reshape(6)
    osc_damping = np.asarray(args.osc_damping, dtype=np.float64).reshape(6)

    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )

    if bool(args.dry_run):
        rows = []
        planned_waypoint_xyz = [] if args.osc_waypoint_xyz is None else [list(row) for row in args.osc_waypoint_xyz]
        planned_waypoint_rpy = [] if args.osc_waypoint_rpy_deg is None else [list(row) for row in args.osc_waypoint_rpy_deg]
        for iteration in range(int(args.iteration_offset), int(args.iteration_offset) + max(int(args.num_iterations), 1)):
            for spec in conditions:
                rows.append(
                    {
                        "iteration": int(iteration),
                        "condition": spec.label,
                        "condition_key": spec.key,
                        "source_segment": spec.source_segment,
                        "switch_behavior": spec.switch_behavior,
                        "target_osc_segment": spec.target_segment,
                        "source_seed": int(args.source_seed) + int(iteration) * 17,
                        "sim_seed": int(args.sim_seed) + int(iteration) * 1009,
                        "osc_delta_xyz": list(args.osc_delta_xyz),
                        "osc_delta_rpy_deg": list(args.osc_delta_rpy_deg),
                        "osc_waypoint_xyz": planned_waypoint_xyz,
                        "osc_waypoint_rpy_deg": planned_waypoint_rpy,
                        "target_ee_samples": 0,
                        "target_ee_pos_rmse_m": float("nan"),
                        "target_ee_final_error_m": float("nan"),
                        "target_ideal_ee_samples": 0,
                        "target_vs_ideal_ee_pos_rmse_m": float("nan"),
                        "target_vs_ideal_ee_final_error_m": float("nan"),
                        "source_final_q_error_rad": float("nan"),
                        "switch_actual_to_osc_start_pos_error_m": float("nan"),
                        "controller_side_guard": bool(args.controller_side_guard),
                        "controller_guard_velocity_threshold": float(args.controller_guard_velocity_threshold),
                        "tam_embeddings_total": 0,
                        "tam_embeddings_sent": 0,
                    }
                )
        _write_summary_files(run_dir, rows)
        _write_aggregate_files(run_dir, rows)
        print(f"[dry-run] Sim references and planned table saved to {run_dir}")
        return run_dir

    if sim_backend == "batched":
        return _run_batched_table(args, conditions, run_dir)

    need_tam = any(spec.tam_on_source or spec.tam_on_target for spec in conditions)
    tam_state = _build_tam_state(args) if need_tam else None
    if need_tam and int(args.adaptor_seq_length) <= 0:
        sim_inf = getattr(tam_state.runtime, "inf", None) if tam_state is not None else None
        args.adaptor_seq_length = int(getattr(sim_inf, "adaptor_seq_length", 1) or 1)
        print(f"[sim] Inferred adaptor_seq_length={int(args.adaptor_seq_length)} from checkpoint.")
    elif int(args.adaptor_seq_length) <= 0:
        args.adaptor_seq_length = 1

    rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    n_iter = max(int(args.num_iterations), 1)
    iteration_offset = int(args.iteration_offset)
    for local_iteration, iteration in enumerate(range(iteration_offset, iteration_offset + n_iter)):
        iter_dir = run_dir / f"iter_{iteration:03d}" if n_iter > 1 else run_dir
        iter_dir.mkdir(parents=True, exist_ok=True)
        ref = _randomized_iteration_reference(
            args=args,
            iteration=iteration,
            initial_q=initial_q,
        )
        np.savez(
            iter_dir / "reference.npz",
            source_t=ref.source_t,
            source_q=ref.source_q,
            source_dq=ref.source_dq,
            target_t=ref.target_t,
            target_pos=ref.target_pos,
            target_quat=ref.target_quat,
            initial_q=initial_q,
            osc_start_pos=ref.osc_start_pos,
            osc_start_quat=ref.osc_start_quat,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            osc_stiffness=osc_stiffness,
            osc_damping=osc_damping,
            osc_nullspace_stiffness=np.asarray(float(args.osc_nullspace_stiffness), dtype=np.float64),
            source_amp_deg=np.asarray(ref.source_amp_deg, dtype=np.float32),
            source_cycles=np.asarray(ref.source_cycles, dtype=np.int32),
            osc_delta_xyz=np.asarray(ref.osc_delta_xyz, dtype=np.float32),
            osc_delta_rpy_deg=np.asarray(ref.osc_delta_rpy_deg, dtype=np.float32),
            osc_waypoint_xyz=np.asarray(ref.osc_waypoint_xyz, dtype=np.float32),
            osc_waypoint_rpy_deg=np.asarray(ref.osc_waypoint_rpy_deg, dtype=np.float32),
            source_seed=np.asarray(ref.source_seed, dtype=np.int64),
            sim_seed=np.asarray(ref.sim_seed, dtype=np.int64),
        )
        ideal_params, perturbed_params, setup_meta = _sample_params(
            xml_path=args.xml.expanduser().resolve(),
            profile_table=args.profile_table.expanduser().resolve(),
            profile_key=args.profile_key,
            seed=int(ref.sim_seed),
            dt_s=float(args.dt),
            param_debug_mode=str(args.param_debug_mode),
        )
        setup_row = {
            "local_iteration": int(local_iteration),
            "iteration": int(iteration),
            "source_seed": int(ref.source_seed),
            "sim_seed": int(ref.sim_seed),
            "osc_delta_xyz": list(ref.osc_delta_xyz),
            "osc_delta_rpy_deg": list(ref.osc_delta_rpy_deg),
            "osc_waypoint_xyz": [list(row) for row in ref.osc_waypoint_xyz],
            "osc_waypoint_rpy_deg": [list(row) for row in ref.osc_waypoint_rpy_deg],
            "source_amp_deg": list(ref.source_amp_deg),
            "source_cycles": list(ref.source_cycles),
            "resolved_history_torque_mode": str(
                getattr(args, "resolved_history_torque_mode", "") or ""
            ),
            **setup_meta,
        }
        setup_rows.append(setup_row)
        (iter_dir / "sim_setup.json").write_text(
            json.dumps(setup_row, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )

        iteration_meta = {
            "iteration": int(iteration),
            "source_seed": int(ref.source_seed),
            "sim_seed": int(ref.sim_seed),
            "osc_delta_xyz": list(ref.osc_delta_xyz),
            "osc_delta_rpy_deg": list(ref.osc_delta_rpy_deg),
            "osc_waypoint_xyz": [list(row) for row in ref.osc_waypoint_xyz],
            "osc_waypoint_rpy_deg": [list(row) for row in ref.osc_waypoint_rpy_deg],
        }
        print(
            f"[sim-iter {iteration}] source_seed={ref.source_seed}, sim_seed={ref.sim_seed}, "
            f"waypoints={np.asarray(ref.osc_waypoint_xyz)}, "
            f"rpy_deg={np.asarray(ref.osc_waypoint_rpy_deg)}",
            flush=True,
        )
        _simulate_condition(
            args=args,
            spec=IDEAL_SPEC,
            run_dir=iter_dir,
            condition_dir_name="ideal_model",
            iteration_meta=iteration_meta,
            source_t=ref.source_t,
            source_q=ref.source_q,
            source_dq=ref.source_dq,
            target_t=ref.target_t,
            target_pos=ref.target_pos,
            target_quat=ref.target_quat,
            initial_q=initial_q,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            osc_stiffness=osc_stiffness,
            osc_damping=osc_damping,
            osc_nullspace_stiffness=float(args.osc_nullspace_stiffness),
            rollout_params=ideal_params,
            tam_state=None,
        )
        ideal_source_npz = np.load(iter_dir / "ideal_model" / "source_log.npz", allow_pickle=False)
        ideal_source_log_np = {key: np.asarray(ideal_source_npz[key]) for key in ideal_source_npz.files}
        ideal_npz = np.load(iter_dir / "ideal_model" / "target_log.npz", allow_pickle=False)
        ideal_target_log_np = {key: np.asarray(ideal_npz[key]) for key in ideal_npz.files}

        iter_rows: list[dict[str, Any]] = []
        for spec in conditions:
            print(f"[sim-iter {iteration}][condition] {spec.label}", flush=True)
            row = _simulate_condition(
                args=args,
                spec=spec,
                run_dir=iter_dir,
                iteration_meta=iteration_meta,
                ideal_source_log_np=ideal_source_log_np,
                ideal_target_log_np=ideal_target_log_np,
                source_t=ref.source_t,
                source_q=ref.source_q,
                source_dq=ref.source_dq,
                target_t=ref.target_t,
                target_pos=ref.target_pos,
                target_quat=ref.target_quat,
                initial_q=initial_q,
                joint_kp=joint_kp,
                joint_kd=joint_kd,
                osc_stiffness=osc_stiffness,
                osc_damping=osc_damping,
                osc_nullspace_stiffness=float(args.osc_nullspace_stiffness),
                rollout_params=perturbed_params,
                tam_state=tam_state,
            )
            rows.append(row)
            iter_rows.append(row)
            _write_summary_files(iter_dir, iter_rows)
            _write_aggregate_files(iter_dir, iter_rows)
            _write_summary_files(run_dir, rows)
            _write_aggregate_files(run_dir, rows)
            print(
                f"[sim-iter {iteration}][condition] {spec.label}: "
                f"cmd_rmse={row['target_ee_pos_rmse_m']:.6f}m, "
                f"ideal_rmse={row['target_vs_ideal_ee_pos_rmse_m']:.6f}m, "
                f"tam_sent={row['tam_embeddings_sent']}",
                flush=True,
            )

    (run_dir / "sim_setup.json").write_text(
        json.dumps(setup_rows, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _write_summary_files(run_dir, rows)
    _write_aggregate_files(run_dir, rows)
    print(f"[done] Sim logs and table saved to {run_dir}")
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


__all__ = [
    "OnlineTamSimState",
    "build_arg_parser",
    "main",
    "run",
]
