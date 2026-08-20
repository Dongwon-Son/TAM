"""MuJoCo "hardware-in-the-loop minus hardware" backend for the TAM NUC bridge.

Runs the RCS torque-law replica (:mod:`simadaptor.deploy.rcs_controller_replica`)
on a MuJoCo Panda at 1 kHz wall-clock, with the RCS-fork C++ ``TamHook``
(``rcs_panda._core.hw.TamHook`` / ``rcs_fr3``) in the loop when available, and
serves the same backend interface as the RCS hardware backend
(:class:`rcs_tam.backend.BridgeBackend`).  Together with
:class:`rcs_tam.bridge.TamBridge` this lets the unchanged workstation
stack (``mapping_server.py``, ``HistoryControllerClient``, launchers) run
end-to-end without a robot: protocol, timing, embedding flow, resets and
logging are exercised exactly as on the NUC.

Torque bookkeeping mirrors the RCS fork:

    tau_base (gravity-free)  = RCS law(q, dq, target)
    delta                    = TamHook.apply(...)               (0 without hook)
    tau_applied (gravity-free)= RCS tail(tau_base + delta)      (rate limit, clamp,
                                                                  libfranka low-pass)
    plant torque             = actuator_model(tau_applied + g_ideal(q))

``rcs_tam`` (the RCS fork's ``extensions/rcs_tam``) must be importable
(``pip install -e extensions/rcs_tam``, or ``--rcs-root`` in the launcher).
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from simadaptor.deploy.rcs_controller_replica import (
    RcsControllerReplica,
    RcsMujocoStepper,
    RcsReplicaConfig,
    quat_normalize,
)

try:  # robot-control-stack (tam branch) extensions/rcs_tam
    from rcs_tam.backend import BridgeBackend, UnsupportedCommand, history_rows_dict_to_samples
except ModuleNotFoundError:  # pragma: no cover - only for import-time friendliness
    BridgeBackend = object  # type: ignore[assignment,misc]

    class UnsupportedCommand(Exception):  # type: ignore[no-redef]
        pass

    history_rows_dict_to_samples = None  # type: ignore[assignment]

HOME_Q = np.asarray([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483])


def _arm_indices(model: Any, dof: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """qpos/qvel indices of the first ``dof`` arm hinge joints (``panda_joint*`` first, then actuated hinges)."""
    import mujoco

    joint_ids = [i for i in range(model.njnt) if (model.joint(i).name or "").startswith("panda_joint")]
    if len(joint_ids) < dof:
        seen: set[int] = set()
        joint_ids = []
        for actuator_idx in range(int(model.nu)):
            joint_id = int(model.actuator_trnid[actuator_idx, 0])
            if joint_id < 0 or joint_id in seen or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            seen.add(joint_id)
            joint_ids.append(joint_id)
    if len(joint_ids) < dof:
        joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(joint_ids) < dof:
        raise ValueError(f"model has only {len(joint_ids)} hinge joints; need {dof}")
    joint_ids = joint_ids[:dof]
    qpos_idx = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32)
    qvel_idx = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32)
    return qpos_idx, qvel_idx


def _set_arm_state(model: Any, data: Any, qpos_idx: np.ndarray, qvel_idx: np.ndarray, q: np.ndarray, dq: np.ndarray) -> None:
    import mujoco

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
    data.qvel[qvel_idx] = np.asarray(dq, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(model, data)


class _PyHistory:
    """Minimal Python stand-in for ``TamHook`` when the RCS extension is unavailable (delta = 0)."""

    def __init__(self, max_rows: int = 4096) -> None:
        self._rows: collections.deque = collections.deque(maxlen=int(max_rows))
        self._lock = threading.Lock()
        self.embedding: Optional[np.ndarray] = None
        self.embedding_seq = 0
        self.enabled = False
        self.gravity_flag = True
        self.t0 = time.perf_counter()
        self.start_count = 0
        self.rows_total = 0

    def on_control_start(self) -> None:
        with self._lock:
            self._rows.clear()
            self.embedding = None
            self.embedding_seq = 0
            self.t0 = time.perf_counter()
            self.start_count += 1

    def load_adaptor(self, path: str) -> bool:
        return False

    def set_embedding(self, z: np.ndarray) -> None:
        with self._lock:
            self.embedding = np.asarray(z, dtype=float)
            self.embedding_seq += 1

    def embedding_seq_(self) -> int:
        with self._lock:
            return self.embedding_seq

    def apply(self, dt: float, q, dq, tau_base, gravity, tau_cmd, tau_meas) -> np.ndarray:
        with self._lock:
            self._rows.append(
                {
                    "t": time.perf_counter() - self.t0,
                    "q": np.asarray(q, float).copy(),
                    "dq": np.asarray(dq, float).copy(),
                    "tau_base": np.asarray(tau_base, float).copy(),
                    "tau_adaptor_delta": np.zeros(7),
                    "tau_applied": np.zeros(7),
                    "tau_commanded": np.asarray(tau_cmd, float).copy(),
                    "tau_measured": np.asarray(tau_meas, float).copy(),
                    "gravity": np.asarray(gravity, float).copy(),
                    "history_embedding_seq": self.embedding_seq,
                    "adaptor_active": False,
                    "valid_for_history": bool(np.max(np.abs(tau_base)) > 1e-5),
                    "synthetic_padding": False,
                    "publish_ready": False,
                    "sample_dt_sec": float(dt),
                }
            )
            self.rows_total += 1
        return np.zeros(7)

    def finalize_row(self, tau_applied) -> None:
        with self._lock:
            if not self._rows:
                return
            row = self._rows[-1]
            row["tau_applied"] = np.asarray(tau_applied, float).copy()
            row["valid_for_history"] = bool(np.max(np.abs(row["tau_applied"])) > 1e-5)
            row["publish_ready"] = True

    def get_history(self, max_rows: int) -> Dict[str, np.ndarray]:
        with self._lock:
            rows = [r for r in list(self._rows)[-int(max_rows):] if r["publish_ready"]]
        keys = ["t", "q", "dq", "tau_base", "tau_adaptor_delta", "tau_applied", "tau_commanded", "tau_measured",
                "gravity", "history_embedding_seq", "adaptor_active", "valid_for_history", "synthetic_padding",
                "sample_dt_sec"]
        if not rows:
            return {"t": np.zeros(0)}
        return {k: np.asarray([r[k] for r in rows]) for k in keys}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "loaded": False,
                "enabled": self.enabled,
                "ideal_model_has_gravity": self.gravity_flag,
                "embedding_seq": self.embedding_seq,
                "history_size": len(self._rows),
                "rows_total": self.rows_total,
                "start_count": self.start_count,
                "last_skip_reason": "not_loaded",
                "adaptor_forward_dt_ms": 0.0,
            }


class RcsMujocoBackend(BridgeBackend):  # type: ignore[misc]
    name = "rcs_mujoco"

    def __init__(
        self,
        *,
        xml_path: str,
        hook: Any = None,
        replica_cfg: Optional[RcsReplicaConfig] = None,
        initial_q: Optional[np.ndarray] = None,
        dt: float = 1e-3,
        realtime: bool = True,
        fk_site: str = "gripper",
        plant_modifier: Optional[Callable[[Any], None]] = None,
        actuator_model: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        default_control_mode: str = "joint",
        log=print,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self._log = log
        self.dt = float(dt)
        self.realtime = bool(realtime)
        self.initial_q = HOME_Q.copy() if initial_q is None else np.asarray(initial_q, dtype=float).reshape(7)
        self.hook = hook if hook is not None else _PyHistory()
        self.hook_is_native = hook is not None
        self.replica = RcsControllerReplica(replica_cfg or RcsReplicaConfig())
        self.control_mode = default_control_mode

        self.controller_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.controller_model.opt.timestep = self.dt
        self.controller_model.body_gravcomp[:] = 0.0
        self.controller_data = mujoco.MjData(self.controller_model)
        self.plant_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.plant_model.opt.timestep = self.dt
        self.plant_model.body_gravcomp[:] = 0.0
        if plant_modifier is not None:
            plant_modifier(self.plant_model)
        self.plant_data = mujoco.MjData(self.plant_model)
        self.actuator_model = actuator_model

        self._set_arm_state = _set_arm_state
        self.qpos_idx, self.qvel_idx = _arm_indices(self.plant_model, dof=7)
        ctrl_qpos_idx, ctrl_qvel_idx = _arm_indices(self.controller_model, dof=7)
        site_id = int(mujoco.mj_name2id(self.controller_model, mujoco.mjtObj.mjOBJ_SITE, str(fk_site)))
        if site_id < 0:
            raise ValueError(f"site {fk_site!r} not found in {xml_path}")
        self.stepper = RcsMujocoStepper(
            replica=self.replica,
            controller_model=self.controller_model,
            controller_data=self.controller_data,
            qpos_idx=ctrl_qpos_idx,
            qvel_idx=ctrl_qvel_idx,
            site_id=site_id,
        )
        self._lock = threading.RLock()
        self._pending_joint_goal: Optional[np.ndarray] = None
        self._pending_pose_goal: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.sim_time = 0.0
        self.ticks = 0
        self.overruns = 0
        self.last_delta = np.zeros(7)
        self.last_tau_applied = np.zeros(7)
        self.last_q = self.initial_q.copy()
        self._reset_sim(mode=default_control_mode)

    # ------------------------------------------------------------------ #
    # sim thread
    # ------------------------------------------------------------------ #
    def _reset_sim(self, mode: str) -> None:
        with self._lock:
            self._set_arm_state(self.plant_model, self.plant_data, self.qpos_idx, self.qvel_idx, self.initial_q, np.zeros(7))
            self.replica.start("joint" if mode == "joint" else "osc")
            self.control_mode = mode
            self.hook.on_control_start()
            self._pending_joint_goal = None
            self._pending_pose_goal = None
            self.sim_time = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rcs_mujoco_backend", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        mujoco = self._mujoco
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                q = np.asarray(self.plant_data.qpos[self.qpos_idx], dtype=float)
                dq = np.asarray(self.plant_data.qvel[self.qvel_idx], dtype=float)
                mujoco.mj_forward(self.plant_model, self.plant_data)
                quantities = self.stepper.model_quantities(q)
                gravity = quantities["gravity"]
                if self.ticks > 0:
                    self.replica.tick()
                if self.control_mode == "joint":
                    if self._pending_joint_goal is not None:
                        self.replica.send_joint_goal(self._pending_joint_goal, q)
                        self._pending_joint_goal = None
                    tau_nograv, _info = self.replica.base_torque_joint(q, dq)
                else:
                    if self._pending_pose_goal is not None:
                        pos, quat = self._pending_pose_goal
                        self.replica.send_pose_goal(pos, quat, quantities["ee_pos"], quantities["ee_quat"])
                        self._pending_pose_goal = None
                    tau_nograv, _info = self.replica.base_torque_osc(
                        q,
                        dq,
                        mass_matrix=quantities["mass"],
                        jacobian=quantities["jacobian"],
                        ee_pos=quantities["ee_pos"],
                        ee_rot=quantities["ee_rot"],
                        ee_quat_wxyz=quantities["ee_quat"],
                    )
                tau_measured = self.last_tau_applied + gravity  # crude stand-in for tau_J
                delta = np.asarray(
                    self.hook.apply(self.dt, q, dq, tau_nograv, gravity, self.replica.tau_J_d, tau_measured),
                    dtype=float,
                ).reshape(7)
                tau_applied = self.replica.apply_tail(tau_nograv + delta)
                self.hook.finalize_row(tau_applied)
                tau_plant = tau_applied + gravity
                if self.actuator_model is not None:
                    tau_plant = self.actuator_model(tau_plant, q, dq)
                self.plant_data.ctrl[:7] = tau_plant
                mujoco.mj_step(self.plant_model, self.plant_data)
                self.sim_time += self.dt
                self.ticks += 1
                self.last_delta = delta
                self.last_tau_applied = tau_applied
                self.last_q = q
            if self.realtime:
                next_tick += self.dt
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick - now)
                elif now - next_tick > 0.05:
                    self.overruns += 1
                    next_tick = now

    # ------------------------------------------------------------------ #
    # BridgeBackend
    # ------------------------------------------------------------------ #
    def get_history_samples(self, max_rows: int) -> List[Dict[str, Any]]:
        if history_rows_dict_to_samples is None:
            raise RuntimeError("rcs_tam is not importable (install the RCS fork extension extensions/rcs_tam)")
        return history_rows_dict_to_samples(self.hook.get_history(int(max_rows)))

    def set_embedding(self, embedding: np.ndarray) -> None:
        self.hook.set_embedding(np.asarray(embedding, dtype=float).reshape(-1))

    def get_embedding_seq(self) -> int:
        if self.hook_is_native:
            return int(self.hook.embedding_seq())
        return int(self.hook.embedding_seq_())

    def enable_adaptor(self, enabled: bool) -> None:
        if self.hook_is_native:
            self.hook.enable(bool(enabled))
        else:
            self.hook.enabled = bool(enabled)

    def adaptor_enabled(self) -> bool:
        if self.hook_is_native:
            return bool(self.hook.enabled())
        return bool(self.hook.enabled)

    def load_adaptor_path(self, path: str) -> bool:
        return bool(self.hook.load_adaptor(str(path)))

    def set_ideal_model_has_gravity(self, enabled: bool) -> None:
        if self.hook_is_native:
            self.hook.set_ideal_model_has_gravity(bool(enabled))
        else:
            self.hook.gravity_flag = bool(enabled)

    def ideal_model_has_gravity(self) -> bool:
        if self.hook_is_native:
            return bool(self.hook.ideal_model_has_gravity())
        return bool(self.hook.gravity_flag)

    def apply_actuation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        resp: Dict[str, Any] = {}
        with self._lock:
            if "control_mode" in payload:
                mode = "cartesian" if str(payload["control_mode"]).lower().startswith("cart") else "joint"
                if mode != self.control_mode:
                    self.replica.start("joint" if mode == "joint" else "osc")
                    self.hook.on_control_start()
                    self.control_mode = mode
                    self._pending_joint_goal = None
                    self._pending_pose_goal = None
            if "target_q" in payload:
                if self.control_mode != "joint":
                    self.replica.start("joint")
                    self.hook.on_control_start()
                    self.control_mode = "joint"
                self._pending_joint_goal = np.asarray(payload["target_q"], dtype=float).reshape(7)
            if "target_position" in payload or "target_orientation" in payload:
                if self.control_mode != "cartesian":
                    self.replica.start("osc")
                    self.hook.on_control_start()
                    self.control_mode = "cartesian"
                pos, quat = self._current_pose_goal()
                if "target_position" in payload:
                    pos = np.asarray(payload["target_position"], dtype=float).reshape(3)
                if "target_orientation" in payload:
                    q_xyzw = np.asarray(payload["target_orientation"], dtype=float).reshape(4)
                    quat = quat_normalize(np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]))
                self._pending_pose_goal = (pos, quat)
        unsupported = sorted(k for k in payload if k in ("filter", "feedforward", "damping_ratio", "stiffness", "damping",
                                                        "cartesian_stiffness", "cartesian_damping",
                                                        "cartesian_nullspace_stiffness", "target_q_nullspace"))
        if unsupported:
            resp["unsupported_keys"] = unsupported
        return resp

    def _current_pose_goal(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending_pose_goal is not None:
            return self._pending_pose_goal
        quantities = self.stepper.model_quantities(np.asarray(self.plant_data.qpos[self.qpos_idx], dtype=float))
        return quantities["ee_pos"], quantities["ee_quat"]

    def neutralize_remote_motion(self) -> None:
        with self._lock:
            q = np.asarray(self.plant_data.qpos[self.qpos_idx], dtype=float)
            if self.control_mode != "joint":
                self.replica.start("joint")
                self.hook.on_control_start()
                self.control_mode = "joint"
            self._pending_joint_goal = q

    def reset(self) -> None:
        self._reset_sim(mode="joint")

    def status(self) -> Dict[str, Any]:
        st = dict(self.hook.status())
        st.pop("torque_limits", None)
        out = {f"tam_{k}": v for k, v in st.items()}
        out.update(
            {
                "sim_time": float(self.sim_time),
                "sim_ticks": int(self.ticks),
                "sim_overruns": int(self.overruns),
                "sim_control_mode": self.control_mode,
                "sim_last_delta_max": float(np.max(np.abs(self.last_delta))),
                "native_hook": bool(self.hook_is_native),
            }
        )
        return out

    def close(self) -> None:
        self.stop()


__all__ = ["RcsMujocoBackend"]
