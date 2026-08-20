"""MuJoCo replica of the robot-control-stack (RCS) Franka low-level controllers.

This module reproduces, in numpy, the nominal-torque (tau^0) generators that
``rcs_fr3``/``rcs_panda`` run inside ``hw::Franka``'s 1 kHz libfranka torque
callback (``extensions/rcs_fr3/src/hw/Franka.cpp`` @ RCS ``main``, 2026-08):

* ``joint_controller()``   -- ``tau = Kp*(q_d - q) - Kd*dq`` with joint-limit
  hard zeroing, deoxys ``joint-impedance-controller.yml`` gains.
* ``osc()``                -- deoxys-derived operational-space controller with
  residual mass, SVD pseudo-inverse task inertias, error deadbands, unit-gain
  nullspace pull toward a static posture, joint-limit avoidance potentials and
  hard zeroing.
* ``LinearJointPositionTrajInterpolator`` / ``LinearPoseTrajInterpolator`` --
  the setpoint hand-off used by ``controller_set_joint_position`` /
  ``osc_set_cartesian_position`` (hard-coded ``policy_rate=20``,
  ``traj_rate=500``, ``traj_interpolation_time_fraction=1.0``).
* The RCS torque tail: ``franka::limitRate(kMaxTorqueRate)`` followed by the
  ``TorqueSafetyGuardFn`` clamp to ``FrankaConfig::torque_limit`` (default 5 Nm).
* The libfranka command tail applied by ``franka::Robot::control`` with default
  arguments (``limit_rate=true``, ``cutoff_frequency=100 Hz``): first-order
  low-pass on the commanded torque followed by rate limiting.  pandapy_dw uses
  the same defaults (``robot_->control(control_callback)``).

RCS commands are gravity-free (libfranka adds its model gravity on the robot);
the sim helpers therefore return gravity-free torques and the caller adds the
ideal-model gravity to obtain TAM's model-space ``tau^0``.

The TAM hook of the RCS fork sits between the RCS law and the RCS tail, which
is why :class:`RcsControllerReplica` exposes ``base_torque_*`` and
``apply_tail`` separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# RCS / deoxys / libfranka constants (Franka.h / Franka.cpp / libfranka)
# --------------------------------------------------------------------------- #

RCS_DEFAULT_JOINT_KP = (100.0, 100.0, 100.0, 100.0, 75.0, 150.0, 50.0)
RCS_DEFAULT_JOINT_KD = (20.0, 20.0, 20.0, 20.0, 7.5, 15.0, 5.0)
RCS_DEFAULT_KP_P = (150.0, 150.0, 150.0)
RCS_DEFAULT_KP_R = 250.0
RCS_DEFAULT_TORQUE_LIMIT_NM = 5.0
RCS_DEFAULT_POLICY_RATE_HZ = 20
RCS_DEFAULT_TRAJ_RATE_HZ = 500
RCS_DEFAULT_TRAJ_TIME_FRACTION = 1.0
RCS_ACTION_DEDUP_ATOL = 1e-3  # RobotWrapper.action(): np.allclose(atol=1e-3, rtol=0)
RCS_OSC_STATIC_Q_TASK = (
    0.09017809387254755,
    -0.9824203501652151,
    0.030509718397568178,
    -2.694229634937343,
    0.057700675144720104,
    1.860298714876101,
    0.8713759453244422,
)
RCS_OSC_RESIDUAL_MASS_VEC = (0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5)
RCS_JOINT_MAX = (2.8978, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
RCS_JOINT_MIN = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
RCS_OSC_AVOIDANCE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0)
RCS_OSC_POS_DEADBAND_M = 1e-4
RCS_OSC_ORI_DEADBAND = 5e-3
RCS_OSC_PINV_EPS = 0.00025
RCS_JOINT_LIMIT_HARD_ZERO_RAD = 0.1
RCS_JOINT_LIMIT_AVOID_OUTER_RAD = 0.25

LIBFRANKA_DELTA_T = 1e-3
LIBFRANKA_MAX_TORQUE_RATE = 1000.0  # Nm/s, franka::kMaxTorqueRate
LIBFRANKA_DEFAULT_CUTOFF_HZ = 100.0  # franka::kDefaultCutoffFrequency
LIBFRANKA_MAX_CUTOFF_HZ = 1000.0  # franka::kMaxCutoffFrequency (>= disables the filter)


# --------------------------------------------------------------------------- #
# quaternion helpers (wxyz)
# --------------------------------------------------------------------------- #


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.asarray([1.0, 0.0, 0.0, 0.0])


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).reshape(4).copy()
    out[1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Eigen ``Quaternion::slerp`` semantics (shortest arc handled by the caller's flip)."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = float(np.dot(q0, q1))
    abs_d = min(abs(d), 1.0)
    if abs_d >= 1.0 - 1e-9:
        scale0 = 1.0 - t
        scale1 = t
    else:
        theta = np.arccos(abs_d)
        sin_theta = np.sin(theta)
        scale0 = np.sin((1.0 - t) * theta) / sin_theta
        scale1 = np.sin(t * theta) / sin_theta
    if d < 0.0:
        scale1 = -scale1
    return quat_normalize(scale0 * q0 + scale1 * q1)


def quat_from_rotmat(rot: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros((4,), dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(9))
    return quat


# --------------------------------------------------------------------------- #
# libfranka helpers
# --------------------------------------------------------------------------- #


def libfranka_limit_rate(
    commanded: np.ndarray,
    last_commanded: np.ndarray,
    max_rate: float = LIBFRANKA_MAX_TORQUE_RATE,
    delta_t: float = LIBFRANKA_DELTA_T,
) -> np.ndarray:
    """``franka::limitRate(max_derivatives, commanded, last_commanded)`` for torques."""
    commanded = np.asarray(commanded, dtype=np.float64)
    last_commanded = np.asarray(last_commanded, dtype=np.float64)
    step = float(max_rate) * float(delta_t)
    return last_commanded + np.clip(commanded - last_commanded, -step, step)


def libfranka_lowpass(
    y: np.ndarray,
    y_last: np.ndarray,
    cutoff_hz: float,
    sample_time: float = LIBFRANKA_DELTA_T,
) -> np.ndarray:
    """``franka::lowpassFilter`` (first-order, applied elementwise)."""
    gain = float(sample_time) / (float(sample_time) + 1.0 / (2.0 * np.pi * float(cutoff_hz)))
    return gain * np.asarray(y, dtype=np.float64) + (1.0 - gain) * np.asarray(y_last, dtype=np.float64)


def libfranka_command_tail(
    tau_cmd: np.ndarray,
    last_tau_J_d: np.ndarray,
    *,
    cutoff_hz: Optional[float] = LIBFRANKA_DEFAULT_CUTOFF_HZ,
    limit_rate: bool = True,
) -> np.ndarray:
    """What ``franka::Robot::control(torque_cb, limit_rate, cutoff_frequency)`` does to a torque command."""
    tau = np.asarray(tau_cmd, dtype=np.float64)
    if cutoff_hz is not None and float(cutoff_hz) < LIBFRANKA_MAX_CUTOFF_HZ:
        tau = libfranka_lowpass(tau, last_tau_J_d, float(cutoff_hz))
    if limit_rate:
        tau = libfranka_limit_rate(tau, last_tau_J_d)
    return tau


# --------------------------------------------------------------------------- #
# RCS trajectory interpolators (include/rcs/LinearPoseTrajInterpolator.h)
# --------------------------------------------------------------------------- #


class RcsLinearJointInterpolator:
    """Port of ``rcs::common::LinearJointPositionTrajInterpolator``."""

    def __init__(self) -> None:
        self.dt = 0.0
        self.last_time = 0.0
        self.max_time = 1.0
        self.start_time = 0.0
        self.start = False
        self.first_goal = True
        self.q_start: Optional[np.ndarray] = None
        self.q_goal: Optional[np.ndarray] = None
        self.last_q_t: Optional[np.ndarray] = None
        self.prev_q_goal: Optional[np.ndarray] = None

    def reset(
        self,
        time_sec: float,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        policy_rate: int,
        rate: int,
        traj_interpolator_time_fraction: float,
    ) -> None:
        self.dt = 1.0 / float(rate)
        self.last_time = float(time_sec)
        self.max_time = 1.0 / float(policy_rate) * float(traj_interpolator_time_fraction)
        self.start_time = float(time_sec)
        self.start = False
        if self.first_goal:
            self.q_start = np.asarray(q_start, dtype=np.float64).copy()
            self.prev_q_goal = np.asarray(q_start, dtype=np.float64).copy()
            self.first_goal = False
        else:
            self.prev_q_goal = np.asarray(self.q_goal, dtype=np.float64).copy()
            self.q_start = self.prev_q_goal.copy()
        self.q_goal = np.asarray(q_goal, dtype=np.float64).copy()

    def next_step(self, time_sec: float) -> np.ndarray:
        assert self.q_start is not None and self.q_goal is not None
        if not self.start:
            self.start_time = float(time_sec)
            self.last_q_t = self.q_start.copy()
            self.start = True
        if self.last_time + self.dt <= float(time_sec):
            t = min(max((float(time_sec) - self.start_time) / self.max_time, 0.0), 1.0)
            self.last_q_t = self.q_start + t * (self.q_goal - self.q_start)
            self.last_time = float(time_sec)
        assert self.last_q_t is not None
        return self.last_q_t.copy()


class RcsLinearPoseInterpolator:
    """Port of ``rcs::common::LinearPoseTrajInterpolator`` (quaternions wxyz)."""

    def __init__(self) -> None:
        self.dt = 0.0
        self.last_time = 0.0
        self.max_time = 1.0
        self.start_time = 0.0
        self.start = False
        self.first_goal = True
        self.p_start: Optional[np.ndarray] = None
        self.p_goal: Optional[np.ndarray] = None
        self.q_start: Optional[np.ndarray] = None
        self.q_goal: Optional[np.ndarray] = None
        self.last_p_t: Optional[np.ndarray] = None
        self.last_q_t: Optional[np.ndarray] = None
        self.prev_p_goal: Optional[np.ndarray] = None
        self.prev_q_goal: Optional[np.ndarray] = None

    def reset(
        self,
        time_sec: float,
        p_start: np.ndarray,
        q_start: np.ndarray,
        p_goal: np.ndarray,
        q_goal: np.ndarray,
        policy_rate: int,
        rate: int,
        traj_interpolator_time_fraction: float,
    ) -> None:
        self.dt = 1.0 / float(rate)
        self.last_time = float(time_sec)
        self.max_time = 1.0 / float(policy_rate) * float(traj_interpolator_time_fraction)
        self.start_time = float(time_sec)
        self.start = False
        if self.first_goal:
            self.p_start = np.asarray(p_start, dtype=np.float64).copy()
            self.q_start = quat_normalize(q_start)
            self.prev_p_goal = self.p_start.copy()
            self.prev_q_goal = self.q_start.copy()
            self.first_goal = False
        else:
            self.prev_p_goal = np.asarray(self.p_goal, dtype=np.float64).copy()
            self.prev_q_goal = np.asarray(self.q_goal, dtype=np.float64).copy()
            self.p_start = self.prev_p_goal.copy()
            self.q_start = self.prev_q_goal.copy()
        self.p_goal = np.asarray(p_goal, dtype=np.float64).copy()
        self.q_goal = quat_normalize(q_goal)
        # Flip the sign if the dot product of quaternions is negative.
        if float(np.dot(self.q_goal, self.q_start)) < 0.0:
            self.q_start = -self.q_start

    def next_step(self, time_sec: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.p_start is not None and self.q_start is not None
        assert self.p_goal is not None and self.q_goal is not None
        if not self.start:
            self.start_time = float(time_sec)
            self.last_p_t = self.p_start.copy()
            self.last_q_t = self.q_start.copy()
            self.start = True
        if self.last_time + self.dt <= float(time_sec):
            t = min(max((float(time_sec) - self.start_time) / self.max_time, 0.0), 1.0)
            self.last_p_t = self.p_start + t * (self.p_goal - self.p_start)
            self.last_q_t = quat_slerp(self.q_start, self.q_goal, t)
            self.last_time = float(time_sec)
        assert self.last_p_t is not None and self.last_q_t is not None
        return self.last_p_t.copy(), self.last_q_t.copy()


# --------------------------------------------------------------------------- #
# RCS torque laws
# --------------------------------------------------------------------------- #


def rcs_svd_pinverse(mat: np.ndarray, epsilon: float = RCS_OSC_PINV_EPS) -> np.ndarray:
    """``rcs::hw::PInverse`` (Jacobi SVD, singular values below ``epsilon`` zeroed)."""
    u, s, vh = np.linalg.svd(np.asarray(mat, dtype=np.float64), full_matrices=True)
    s_inv = np.where(s < float(epsilon), 0.0, 1.0 / np.where(s < float(epsilon), 1.0, s))
    S_inv = np.zeros((vh.shape[0], u.shape[1]), dtype=np.float64)
    S_inv[: s.shape[0], : s.shape[0]] = np.diag(s_inv)
    return vh.T @ S_inv @ u.T


def rcs_torque_safety_guard(tau: np.ndarray, torque_limit: np.ndarray) -> np.ndarray:
    """``rcs::hw::TorqueSafetyGuardFn``: per-joint clamp to +-torque_limit."""
    lim = np.asarray(torque_limit, dtype=np.float64)
    return np.clip(np.asarray(tau, dtype=np.float64), -lim, lim)


def rcs_joint_limit_hard_zero(
    tau: np.ndarray,
    q: np.ndarray,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
    margin: float = RCS_JOINT_LIMIT_HARD_ZERO_RAD,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64).copy()
    q = np.asarray(q, dtype=np.float64)
    dist2max = np.asarray(joint_max, dtype=np.float64) - q
    dist2min = q - np.asarray(joint_min, dtype=np.float64)
    tau = np.where((dist2max < margin) & (tau > 0.0), 0.0, tau)
    tau = np.where((dist2min < margin) & (tau < 0.0), 0.0, tau)
    return tau


def rcs_joint_controller_law(
    *,
    q: np.ndarray,
    dq: np.ndarray,
    q_d: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    joint_min: np.ndarray = RCS_JOINT_MIN,
    joint_max: np.ndarray = RCS_JOINT_MAX,
) -> np.ndarray:
    """``Franka::joint_controller()`` torque before rate limiting/clamping (gravity-free)."""
    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    tau = np.asarray(kp, dtype=np.float64) * (np.asarray(q_d, dtype=np.float64) - q) - np.asarray(
        kd, dtype=np.float64
    ) * dq
    return rcs_joint_limit_hard_zero(tau, q, joint_min, joint_max)


def rcs_osc_law(
    *,
    q: np.ndarray,
    dq: np.ndarray,
    mass_matrix: np.ndarray,
    jacobian: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    desired_pos: np.ndarray,
    desired_quat_wxyz: np.ndarray,
    kp_p: np.ndarray = RCS_DEFAULT_KP_P,
    kp_r: float = RCS_DEFAULT_KP_R,
    static_q_task: np.ndarray = RCS_OSC_STATIC_Q_TASK,
    residual_mass_vec: np.ndarray = RCS_OSC_RESIDUAL_MASS_VEC,
    joint_min: np.ndarray = RCS_JOINT_MIN,
    joint_max: np.ndarray = RCS_JOINT_MAX,
    avoidance_weights: np.ndarray = RCS_OSC_AVOIDANCE_WEIGHTS,
    pos_deadband: float = RCS_OSC_POS_DEADBAND_M,
    ori_deadband: float = RCS_OSC_ORI_DEADBAND,
    pinv_eps: float = RCS_OSC_PINV_EPS,
    ee_quat_wxyz: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``Franka::osc()`` torque before rate limiting/clamping (gravity-free).

    ``mass_matrix`` is the 7x7 arm inertia from the controller model (libfranka
    ``model.mass``), ``jacobian`` the 6x7 base-frame end-effector Jacobian
    (``model.zeroJacobian``; rows = linear, angular), ``ee_rot`` the 3x3 TCP
    rotation in base frame.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    dq = np.asarray(dq, dtype=np.float64).reshape(-1)
    dof = q.shape[0]
    M = np.asarray(mass_matrix, dtype=np.float64).copy() + np.diag(
        np.asarray(residual_mass_vec, dtype=np.float64).reshape(dof)
    )
    jac = np.asarray(jacobian, dtype=np.float64).reshape(6, dof)
    jac_pos = jac[:3]
    jac_ori = jac[3:]

    ee_rot = np.asarray(ee_rot, dtype=np.float64).reshape(3, 3)
    quat_ee = quat_normalize(ee_quat_wxyz) if ee_quat_wxyz is not None else quat_from_rotmat(ee_rot)
    quat_d = quat_normalize(desired_quat_wxyz)
    if float(np.dot(quat_d, quat_ee)) < 0.0:
        quat_ee = -quat_ee

    pos_error = np.asarray(desired_pos, dtype=np.float64).reshape(3) - np.asarray(ee_pos, dtype=np.float64).reshape(3)
    quat_error = quat_mul(quat_conj(quat_d), quat_ee)  # desired.inverse() * current
    ori_error = -ee_rot @ quat_error[1:]

    M_inv = np.linalg.inv(M)
    lambda_inv = jac @ M_inv @ jac.T
    lambda_full = rcs_svd_pinverse(lambda_inv, pinv_eps)
    j_inv = M_inv @ jac.T @ lambda_full
    nullspace = np.eye(dof) - jac.T @ j_inv.T

    lambda_pos = rcs_svd_pinverse(jac_pos @ M_inv @ jac_pos.T, pinv_eps)
    lambda_ori = rcs_svd_pinverse(jac_ori @ M_inv @ jac_ori.T, pinv_eps)

    pos_error = np.where(np.abs(pos_error) < float(pos_deadband), 0.0, pos_error)
    ori_error = np.where(np.abs(ori_error) < float(ori_deadband), 0.0, ori_error)

    kp_p = np.asarray(kp_p, dtype=np.float64).reshape(3)
    kd_p = 2.0 * np.sqrt(kp_p)
    kp_r_vec = np.full(3, float(kp_r))
    kd_r_vec = 2.0 * np.sqrt(kp_r_vec)

    tau_task = jac_pos.T @ (lambda_pos @ (kp_p * pos_error - kd_p * (jac_pos @ dq))) + jac_ori.T @ (
        lambda_ori @ (kp_r_vec * ori_error - kd_r_vec * (jac_ori @ dq))
    )
    tau_null = nullspace @ (np.asarray(static_q_task, dtype=np.float64).reshape(dof) - q)

    joint_max_arr = np.asarray(joint_max, dtype=np.float64).reshape(dof)
    joint_min_arr = np.asarray(joint_min, dtype=np.float64).reshape(dof)
    weights = np.asarray(avoidance_weights, dtype=np.float64).reshape(dof)
    dist2max = joint_max_arr - q
    dist2min = q - joint_min_arr
    avoidance = np.zeros(dof, dtype=np.float64)
    near_max = (dist2max < RCS_JOINT_LIMIT_AVOID_OUTER_RAD) & (dist2max > RCS_JOINT_LIMIT_HARD_ZERO_RAD)
    near_min = (dist2min < RCS_JOINT_LIMIT_AVOID_OUTER_RAD) & (dist2min > RCS_JOINT_LIMIT_HARD_ZERO_RAD)
    avoidance = avoidance + np.where(near_max, -weights * dist2max, 0.0)
    avoidance = avoidance + np.where(near_min, weights * dist2min, 0.0)
    tau_avoid = nullspace @ avoidance

    tau = tau_task + tau_null + tau_avoid
    tau = rcs_joint_limit_hard_zero(tau, q, joint_min_arr, joint_max_arr)
    info = {
        "pos_error": pos_error,
        "ori_error": ori_error,
        "tau_task": tau_task,
        "tau_nullspace": tau_null,
        "tau_avoidance": tau_avoid,
    }
    return tau, info


# --------------------------------------------------------------------------- #
# stateful replica of hw::Franka's async control thread
# --------------------------------------------------------------------------- #


@dataclass
class RcsReplicaConfig:
    dof: int = 7
    joint_kp: tuple[float, ...] = RCS_DEFAULT_JOINT_KP
    joint_kd: tuple[float, ...] = RCS_DEFAULT_JOINT_KD
    kp_p: tuple[float, float, float] = RCS_DEFAULT_KP_P
    kp_r: float = RCS_DEFAULT_KP_R
    torque_limit: tuple[float, ...] = (RCS_DEFAULT_TORQUE_LIMIT_NM,) * 7
    policy_rate_hz: int = RCS_DEFAULT_POLICY_RATE_HZ
    traj_rate_hz: int = RCS_DEFAULT_TRAJ_RATE_HZ
    traj_time_fraction: float = RCS_DEFAULT_TRAJ_TIME_FRACTION
    action_dedup_atol: Optional[float] = RCS_ACTION_DEDUP_ATOL
    static_q_task: tuple[float, ...] = RCS_OSC_STATIC_Q_TASK
    residual_mass_vec: tuple[float, ...] = RCS_OSC_RESIDUAL_MASS_VEC
    joint_min: tuple[float, ...] = RCS_JOINT_MIN
    joint_max: tuple[float, ...] = RCS_JOINT_MAX
    avoidance_weights: tuple[float, ...] = RCS_OSC_AVOIDANCE_WEIGHTS
    pos_deadband: float = RCS_OSC_POS_DEADBAND_M
    ori_deadband: float = RCS_OSC_ORI_DEADBAND
    pinv_eps: float = RCS_OSC_PINV_EPS
    rcs_limit_rate: bool = True
    libfranka_cutoff_hz: Optional[float] = LIBFRANKA_DEFAULT_CUTOFF_HZ
    libfranka_limit_rate: bool = True
    control_dt: float = LIBFRANKA_DELTA_T

    def as_dict(self) -> dict[str, Any]:
        return {
            "dof": int(self.dof),
            "joint_kp": [float(x) for x in self.joint_kp],
            "joint_kd": [float(x) for x in self.joint_kd],
            "kp_p": [float(x) for x in self.kp_p],
            "kp_r": float(self.kp_r),
            "torque_limit": [float(x) for x in self.torque_limit],
            "policy_rate_hz": int(self.policy_rate_hz),
            "traj_rate_hz": int(self.traj_rate_hz),
            "traj_time_fraction": float(self.traj_time_fraction),
            "action_dedup_atol": None if self.action_dedup_atol is None else float(self.action_dedup_atol),
            "static_q_task": [float(x) for x in self.static_q_task],
            "residual_mass_vec": [float(x) for x in self.residual_mass_vec],
            "joint_min": [float(x) for x in self.joint_min],
            "joint_max": [float(x) for x in self.joint_max],
            "avoidance_weights": [float(x) for x in self.avoidance_weights],
            "pos_deadband": float(self.pos_deadband),
            "ori_deadband": float(self.ori_deadband),
            "pinv_eps": float(self.pinv_eps),
            "rcs_limit_rate": bool(self.rcs_limit_rate),
            "libfranka_cutoff_hz": None if self.libfranka_cutoff_hz is None else float(self.libfranka_cutoff_hz),
            "libfranka_limit_rate": bool(self.libfranka_limit_rate),
            "control_dt": float(self.control_dt),
        }


@dataclass
class RcsReplicaStepInfo:
    mode: str
    goal_sent: bool
    q_d: Optional[np.ndarray] = None
    pos_d: Optional[np.ndarray] = None
    quat_d: Optional[np.ndarray] = None
    tau_pre_tail: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)


class RcsControllerReplica:
    """Stateful replica of one ``hw::Franka`` async control thread.

    Life cycle mirrors RCS: :meth:`start` corresponds to spawning the control
    thread (``controller_time=0``, fresh interpolator, ``tau_J_d=0``);
    :meth:`send_joint_goal` / :meth:`send_pose_goal` correspond to the Python
    ``set_joint_position`` / ``set_cartesian_position`` calls (with the
    ``RobotWrapper`` duplicate-action skip); :meth:`base_torque` is the RCS law
    evaluated inside the 1 kHz callback; :meth:`apply_tail` is everything after
    the TAM hook (RCS rate limit + clamp, then libfranka's low-pass + rate limit).
    """

    def __init__(self, cfg: RcsReplicaConfig) -> None:
        self.cfg = cfg
        self.mode: Optional[str] = None
        self.controller_time = 0.0
        self.tau_J_d = np.zeros(cfg.dof, dtype=np.float64)
        self._joint_interp = RcsLinearJointInterpolator()
        self._pose_interp = RcsLinearPoseInterpolator()
        self._last_action: Optional[np.ndarray] = None
        self._q_d_last: Optional[np.ndarray] = None
        self._pose_d_last: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.num_goals_sent = 0
        self.num_goals_skipped = 0

    # -- thread life cycle -------------------------------------------------- #
    def start(self, mode: str, *, tau_J_d: Optional[np.ndarray] = None) -> None:
        if mode not in ("joint", "osc"):
            raise ValueError(f"mode must be 'joint' or 'osc', got {mode!r}")
        self.mode = mode
        self.controller_time = 0.0
        self.tau_J_d = (
            np.zeros(self.cfg.dof, dtype=np.float64)
            if tau_J_d is None
            else np.asarray(tau_J_d, dtype=np.float64).reshape(self.cfg.dof).copy()
        )
        self._joint_interp = RcsLinearJointInterpolator()
        self._pose_interp = RcsLinearPoseInterpolator()
        self._last_action = None
        self._q_d_last = None
        self._pose_d_last = None

    def stop(self) -> None:
        self.mode = None

    # -- Python-side setpoint calls ---------------------------------------- #
    def _dedup(self, action: np.ndarray) -> bool:
        """Return True when RobotWrapper.action() would forward this action."""
        atol = self.cfg.action_dedup_atol
        if atol is None or self._last_action is None:
            self._last_action = np.asarray(action, dtype=np.float64).copy()
            return True
        forward = not np.allclose(action, self._last_action, atol=float(atol), rtol=0.0)
        self._last_action = np.asarray(action, dtype=np.float64).copy()
        return forward

    def send_joint_goal(self, q_goal: np.ndarray, q_current: np.ndarray) -> bool:
        assert self.mode == "joint", "send_joint_goal requires the joint controller"
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(self.cfg.dof)
        if not self._dedup(q_goal):
            self.num_goals_skipped += 1
            return False
        self._joint_interp.reset(
            self.controller_time,
            np.asarray(q_current, dtype=np.float64).reshape(self.cfg.dof),
            q_goal,
            int(self.cfg.policy_rate_hz),
            int(self.cfg.traj_rate_hz),
            float(self.cfg.traj_time_fraction),
        )
        self.num_goals_sent += 1
        return True

    def send_pose_goal(
        self,
        pos_goal: np.ndarray,
        quat_goal_wxyz: np.ndarray,
        pos_current: np.ndarray,
        quat_current_wxyz: np.ndarray,
    ) -> bool:
        assert self.mode == "osc", "send_pose_goal requires the OSC controller"
        pos_goal = np.asarray(pos_goal, dtype=np.float64).reshape(3)
        quat_goal = quat_normalize(quat_goal_wxyz)
        # RCS TQuat actions are [xyz, quat]; dedup on that vector.
        if not self._dedup(np.concatenate([pos_goal, quat_goal])):
            self.num_goals_skipped += 1
            return False
        self._pose_interp.reset(
            self.controller_time,
            np.asarray(pos_current, dtype=np.float64).reshape(3),
            quat_normalize(quat_current_wxyz),
            pos_goal,
            quat_goal,
            int(self.cfg.policy_rate_hz),
            int(self.cfg.traj_rate_hz),
            float(self.cfg.traj_time_fraction),
        )
        self.num_goals_sent += 1
        return True

    # -- 1 kHz callback ---------------------------------------------------- #
    def tick(self, period_s: Optional[float] = None) -> float:
        """Advance ``controller_time`` like ``this->controller_time += period.toSec()``."""
        self.controller_time += float(self.cfg.control_dt if period_s is None else period_s)
        return self.controller_time

    def base_torque_joint(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, RcsReplicaStepInfo]:
        assert self.mode == "joint"
        if self._joint_interp.q_goal is None:
            q_d = np.asarray(q, dtype=np.float64).reshape(self.cfg.dof).copy()
        else:
            q_d = self._joint_interp.next_step(self.controller_time)
        tau = rcs_joint_controller_law(
            q=q,
            dq=dq,
            q_d=q_d,
            kp=np.asarray(self.cfg.joint_kp, dtype=np.float64),
            kd=np.asarray(self.cfg.joint_kd, dtype=np.float64),
            joint_min=np.asarray(self.cfg.joint_min, dtype=np.float64),
            joint_max=np.asarray(self.cfg.joint_max, dtype=np.float64),
        )
        return tau, RcsReplicaStepInfo(mode="joint", goal_sent=self._joint_interp.q_goal is not None, q_d=q_d, tau_pre_tail=tau)

    def base_torque_osc(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        *,
        mass_matrix: np.ndarray,
        jacobian: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        ee_quat_wxyz: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, RcsReplicaStepInfo]:
        assert self.mode == "osc"
        if self._pose_interp.p_goal is None:
            pos_d = np.asarray(ee_pos, dtype=np.float64).reshape(3).copy()
            quat_d = quat_normalize(ee_quat_wxyz) if ee_quat_wxyz is not None else quat_from_rotmat(ee_rot)
        else:
            pos_d, quat_d = self._pose_interp.next_step(self.controller_time)
        tau, info = rcs_osc_law(
            q=q,
            dq=dq,
            mass_matrix=mass_matrix,
            jacobian=jacobian,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            desired_pos=pos_d,
            desired_quat_wxyz=quat_d,
            kp_p=np.asarray(self.cfg.kp_p, dtype=np.float64),
            kp_r=float(self.cfg.kp_r),
            static_q_task=np.asarray(self.cfg.static_q_task, dtype=np.float64),
            residual_mass_vec=np.asarray(self.cfg.residual_mass_vec, dtype=np.float64),
            joint_min=np.asarray(self.cfg.joint_min, dtype=np.float64),
            joint_max=np.asarray(self.cfg.joint_max, dtype=np.float64),
            avoidance_weights=np.asarray(self.cfg.avoidance_weights, dtype=np.float64),
            pos_deadband=float(self.cfg.pos_deadband),
            ori_deadband=float(self.cfg.ori_deadband),
            pinv_eps=float(self.cfg.pinv_eps),
            ee_quat_wxyz=ee_quat_wxyz,
        )
        return tau, RcsReplicaStepInfo(
            mode="osc",
            goal_sent=self._pose_interp.p_goal is not None,
            pos_d=pos_d,
            quat_d=quat_d,
            tau_pre_tail=tau,
            extra=info,
        )

    def apply_tail(self, tau_after_hook: np.ndarray) -> np.ndarray:
        """RCS rate limit + clamp, then libfranka's low-pass + rate limit; updates ``tau_J_d``.

        ``tau_after_hook`` is the gravity-free command after the TAM hook
        (``tau_d`` in ``Franka.cpp`` right before ``franka::limitRate``).
        """
        tau = np.asarray(tau_after_hook, dtype=np.float64).reshape(self.cfg.dof)
        if self.cfg.rcs_limit_rate:
            tau = libfranka_limit_rate(tau, self.tau_J_d, LIBFRANKA_MAX_TORQUE_RATE, self.cfg.control_dt)
        tau = rcs_torque_safety_guard(tau, np.asarray(self.cfg.torque_limit, dtype=np.float64))
        tau = libfranka_command_tail(
            tau,
            self.tau_J_d,
            cutoff_hz=self.cfg.libfranka_cutoff_hz,
            limit_rate=bool(self.cfg.libfranka_limit_rate),
        )
        self.tau_J_d = tau.copy()
        return tau


# --------------------------------------------------------------------------- #
# MuJoCo glue used by the source-to-OSC simulator
# --------------------------------------------------------------------------- #


def _dense_mass_matrix(mujoco_mod: Any, model: Any, data: Any, out: np.ndarray) -> None:
    """``mj_fullM`` across MuJoCo API generations: ``(m, d, dst)`` in recent releases, ``(m, dst, d.qM)`` before."""
    try:
        mujoco_mod.mj_fullM(model, data, out)
    except TypeError:
        mujoco_mod.mj_fullM(model, out, data.qM)


class RcsMujocoStepper:
    """Drives :class:`RcsControllerReplica` from MuJoCo controller-model quantities.

    All model quantities (mass matrix, Jacobian, TCP pose, gravity) come from the
    *controller* (ideal) model at the measured joint state, mirroring libfranka's
    nominal ``franka::Model`` on the real robot; the plant is only sampled for
    ``q``/``dq``.  Torques exchanged with the caller are in TAM model space
    (gravity included); the RCS command is recovered by subtracting the ideal
    gravity.
    """

    def __init__(
        self,
        *,
        replica: RcsControllerReplica,
        controller_model: Any,
        controller_data: Any,
        qpos_idx: np.ndarray,
        qvel_idx: np.ndarray,
        site_id: int,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.replica = replica
        self.model = controller_model
        self.data = controller_data
        self.qpos_idx = np.asarray(qpos_idx, dtype=np.int64)
        self.qvel_idx = np.asarray(qvel_idx, dtype=np.int64)
        self.site_id = int(site_id)
        self.dof = int(replica.cfg.dof)
        self._policy_period = 1.0 / float(replica.cfg.policy_rate_hz)
        self._next_send_t: Optional[float] = None
        self._phase_t0 = 0.0
        self._last_gravity = np.zeros(self.dof, dtype=np.float64)
        self._full_M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

    # -- helpers ------------------------------------------------------------ #
    def _forward(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qpos_idx] = np.asarray(q, dtype=np.float64).reshape(self.dof)
        self._mujoco.mj_forward(self.model, self.data)

    def model_quantities(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Gravity, arm mass matrix, TCP Jacobian and TCP pose at ``q`` (zero velocity)."""
        self._forward(q)
        gravity = np.asarray(
            self.data.qfrc_bias[self.qvel_idx] - self.data.qfrc_gravcomp[self.qvel_idx],
            dtype=np.float64,
        )
        _dense_mass_matrix(self._mujoco, self.model, self.data, self._full_M)
        mass = self._full_M[np.ix_(self.qvel_idx, self.qvel_idx)].copy()
        self._mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        jac = np.concatenate([self._jacp[:, self.qvel_idx], self._jacr[:, self.qvel_idx]], axis=0)
        ee_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        ee_rot = np.asarray(self.data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3).copy()
        ee_quat = quat_from_rotmat(ee_rot)
        self._last_gravity = gravity
        return {"gravity": gravity, "mass": mass, "jacobian": jac, "ee_pos": ee_pos, "ee_rot": ee_rot, "ee_quat": ee_quat}

    # -- phases -------------------------------------------------------------- #
    def begin_phase(self, mode: str, t0: float) -> None:
        self.replica.start(mode)
        self._phase_t0 = float(t0)
        self._next_send_t = float(t0)

    def _policy_due(self, t_now: float) -> bool:
        assert self._next_send_t is not None
        if float(t_now) + 1e-9 >= self._next_send_t:
            # Keep the 1/policy_rate cadence anchored to the phase start.
            while self._next_send_t <= float(t_now) + 1e-9:
                self._next_send_t += self._policy_period
            return True
        return False

    def step_joint(
        self,
        *,
        t_now: float,
        q: np.ndarray,
        dq: np.ndarray,
        q_ref: np.ndarray,
    ) -> tuple[np.ndarray, RcsReplicaStepInfo, dict[str, np.ndarray]]:
        """One control tick in joint mode; returns model-space tau^0, step info, model quantities."""
        quantities = self.model_quantities(q)
        # In RCS the first callback has period 0; afterwards controller_time += 1 ms.
        if float(t_now) > self._phase_t0 + 1e-12:
            self.replica.tick()
        goal_sent = False
        if self._policy_due(t_now):
            goal_sent = self.replica.send_joint_goal(q_ref, q)
        tau_nograv, info = self.replica.base_torque_joint(q, dq)
        info.goal_sent = goal_sent
        return tau_nograv + quantities["gravity"], info, quantities

    def step_osc(
        self,
        *,
        t_now: float,
        q: np.ndarray,
        dq: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, RcsReplicaStepInfo, dict[str, np.ndarray]]:
        quantities = self.model_quantities(q)
        if float(t_now) > self._phase_t0 + 1e-12:
            self.replica.tick()
        goal_sent = False
        if self._policy_due(t_now):
            goal_sent = self.replica.send_pose_goal(
                target_pos, target_quat_wxyz, quantities["ee_pos"], quantities["ee_quat"]
            )
        tau_nograv, info = self.replica.base_torque_osc(
            q,
            dq,
            mass_matrix=quantities["mass"],
            jacobian=quantities["jacobian"],
            ee_pos=quantities["ee_pos"],
            ee_rot=quantities["ee_rot"],
            ee_quat_wxyz=quantities["ee_quat"],
        )
        info.goal_sent = goal_sent
        return tau_nograv + quantities["gravity"], info, quantities

    def finalize(self, tau_cmd_model_space: np.ndarray, gravity: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the post-hook tail to a model-space command and return the model-space applied torque."""
        g = self._last_gravity if gravity is None else np.asarray(gravity, dtype=np.float64)
        tau_nograv = np.asarray(tau_cmd_model_space, dtype=np.float64).reshape(self.dof) - g
        return self.replica.apply_tail(tau_nograv) + g


def make_replica_config_from_args(args: Any, dof: int) -> RcsReplicaConfig:
    """Build :class:`RcsReplicaConfig` from the source-to-OSC simulator arguments."""

    def _vec(name: str, default: tuple[float, ...], size: int) -> tuple[float, ...]:
        raw = getattr(args, name, None)
        if raw is None:
            vals = tuple(float(x) for x in default)
        else:
            vals = tuple(float(x) for x in np.asarray(raw, dtype=np.float64).reshape(-1))
        if len(vals) == 1 and size > 1:
            vals = vals * size
        if len(vals) != size:
            raise ValueError(f"--{name.replace('_', '-')} needs {size} values, got {len(vals)}")
        return vals

    def _scalar(name: str, default: float) -> float:
        raw = getattr(args, name, None)
        return float(default) if raw is None else float(raw)

    cutoff = getattr(args, "libfranka_cutoff_hz", LIBFRANKA_DEFAULT_CUTOFF_HZ)
    cutoff_val: Optional[float]
    if cutoff is None:
        cutoff_val = None
    else:
        cutoff_val = float(cutoff)
        if cutoff_val <= 0.0:
            cutoff_val = None
    dedup = getattr(args, "rcs_action_dedup_atol", RCS_ACTION_DEDUP_ATOL)
    return RcsReplicaConfig(
        dof=int(dof),
        joint_kp=_vec("rcs_joint_kp", RCS_DEFAULT_JOINT_KP, dof),
        joint_kd=_vec("rcs_joint_kd", RCS_DEFAULT_JOINT_KD, dof),
        kp_p=tuple(_vec("rcs_kp_p", RCS_DEFAULT_KP_P, 3)),  # type: ignore[arg-type]
        kp_r=_scalar("rcs_kp_r", RCS_DEFAULT_KP_R),
        torque_limit=_vec("rcs_torque_limit", (RCS_DEFAULT_TORQUE_LIMIT_NM,) * dof, dof),
        policy_rate_hz=int(_scalar("rcs_policy_rate_hz", RCS_DEFAULT_POLICY_RATE_HZ)),
        traj_rate_hz=int(_scalar("rcs_traj_rate_hz", RCS_DEFAULT_TRAJ_RATE_HZ)),
        traj_time_fraction=_scalar("rcs_traj_time_fraction", RCS_DEFAULT_TRAJ_TIME_FRACTION),
        action_dedup_atol=None if dedup is None or float(dedup) < 0.0 else float(dedup),
        static_q_task=_vec("rcs_static_q_task", RCS_OSC_STATIC_Q_TASK, dof),
        residual_mass_vec=_vec("rcs_residual_mass_vec", RCS_OSC_RESIDUAL_MASS_VEC, dof),
        joint_min=_vec("rcs_joint_min", RCS_JOINT_MIN, dof),
        joint_max=_vec("rcs_joint_max", RCS_JOINT_MAX, dof),
        avoidance_weights=_vec("rcs_avoidance_weights", RCS_OSC_AVOIDANCE_WEIGHTS, dof),
        pos_deadband=_scalar("rcs_pos_deadband", RCS_OSC_POS_DEADBAND_M),
        ori_deadband=_scalar("rcs_ori_deadband", RCS_OSC_ORI_DEADBAND),
        pinv_eps=_scalar("rcs_pinv_eps", RCS_OSC_PINV_EPS),
        rcs_limit_rate=bool(getattr(args, "rcs_limit_rate", True)),
        libfranka_cutoff_hz=cutoff_val,
        libfranka_limit_rate=bool(getattr(args, "libfranka_limit_rate", True)),
        control_dt=_scalar("dt", LIBFRANKA_DELTA_T),
    )


__all__ = [
    "LIBFRANKA_DEFAULT_CUTOFF_HZ",
    "LIBFRANKA_MAX_TORQUE_RATE",
    "RCS_DEFAULT_JOINT_KD",
    "RCS_DEFAULT_JOINT_KP",
    "RCS_DEFAULT_KP_P",
    "RCS_DEFAULT_KP_R",
    "RCS_DEFAULT_TORQUE_LIMIT_NM",
    "RCS_OSC_STATIC_Q_TASK",
    "RcsControllerReplica",
    "RcsLinearJointInterpolator",
    "RcsLinearPoseInterpolator",
    "RcsMujocoStepper",
    "RcsReplicaConfig",
    "RcsReplicaStepInfo",
    "libfranka_command_tail",
    "libfranka_limit_rate",
    "libfranka_lowpass",
    "make_replica_config_from_args",
    "quat_slerp",
    "rcs_joint_controller_law",
    "rcs_osc_law",
    "rcs_svd_pinverse",
    "rcs_torque_safety_guard",
]
