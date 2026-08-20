from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.core import Tracer as JaxTracer
except ImportError:  # pragma: no cover - JAX is expected in production, but keep import-safe helpers.
    jax = None
    jnp = None
    JaxTracer = ()


@dataclass(frozen=True)
class HistoryRuntimeBundle:
    """Minimal public bundle needed by runtime history updaters."""

    hist_model: Any
    hist_params: Any
    norm_stats: Any
    mjx_model_template: Any
    ideal_model_has_gravity: bool


def _is_jax_value(value: Any) -> bool:
    if jax is None or value is None:
        return False
    if isinstance(value, JaxTracer):
        return True
    array_type = getattr(jax, "Array", None)
    if array_type is not None and isinstance(value, array_type):
        return True
    return hasattr(value, "__jax_array__")


def _array_module(*values: Any):
    if jnp is not None and any(_is_jax_value(v) for v in values):
        return jnp
    return np


def normalize_runtime_history_embedding(history_emb) -> np.ndarray:
    """Normalize runtime embeddings to [1, C] or [1, DoF, C]."""
    emb = np.asarray(history_emb, dtype=np.float32)
    if emb.ndim == 1:
        return emb[None, :]
    if emb.ndim == 2:
        if emb.shape[0] == 1:
            return emb
        return emb[None, ...]
    if emb.ndim == 3:
        if emb.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for runtime history embedding, got shape {emb.shape}."
            )
        return emb
    raise ValueError(
        "Runtime history embedding must normalize to [1, C] or [1, DoF, C], "
        f"got shape {emb.shape}."
    )


def flatten_history_embedding_for_transport(history_emb) -> np.ndarray:
    """Flatten a runtime embedding only at a controller transport boundary."""
    return normalize_runtime_history_embedding(history_emb).reshape(-1)


def prepare_model_space_torque(
    tau,
    gravity=None,
    *,
    ideal_model_has_gravity: bool,
    context: str,
    tau_is_model_space: bool = False,
) -> np.ndarray:
    """Convert raw controller torque to model-space torque when required."""
    xp = _array_module(tau, gravity)
    tau_arr = xp.asarray(tau, dtype=xp.float32)
    if bool(tau_is_model_space) or not bool(ideal_model_has_gravity):
        return tau_arr
    if gravity is None:
        raise ValueError(
            f"{context} requires gravity torque because ideal_model_has_gravity=True."
        )
    gravity_arr = xp.asarray(gravity, dtype=xp.float32)
    if gravity_arr.size == 0:
        raise ValueError(
            f"{context} received an empty gravity torque array while ideal_model_has_gravity=True."
        )
    if gravity_arr.shape != tau_arr.shape:
        raise ValueError(
            f"{context} gravity shape {gravity_arr.shape} does not match tau shape {tau_arr.shape}."
        )
    return tau_arr + gravity_arr


def zero_torque_history_keep_mask(
    raw_tau,
    *,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Return a per-step keep mask from raw controller torque."""
    xp = _array_module(raw_tau)
    tau_arr = xp.asarray(raw_tau, dtype=xp.float32)
    if tau_arr.ndim < 2:
        raise ValueError(f"raw_tau must have rank >= 2 with DoF on the last axis, got {tau_arr.shape}")
    keep = ~xp.all(xp.abs(tau_arr) <= float(threshold), axis=-1)
    return keep.astype(xp.float32)


def mask_zero_torque_history_inputs(
    q,
    qd,
    tau_model,
    *,
    raw_tau=None,
    threshold: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror the adaptor's zero-torque masking at deploy time for history encoding.

    If the raw controller torque is effectively zero for a timestep, zero that
    timestep's q/qd/tau before updating the history encoder so idle controller
    samples do not inject state motion into the history token stream.
    """
    xp = _array_module(q, qd, tau_model, raw_tau)
    q_arr = xp.asarray(q, dtype=xp.float32)
    qd_arr = xp.asarray(qd, dtype=xp.float32)
    tau_arr = xp.asarray(tau_model, dtype=xp.float32)
    ref_tau = tau_arr if raw_tau is None else xp.asarray(raw_tau, dtype=xp.float32)
    if q_arr.shape != qd_arr.shape or q_arr.shape != tau_arr.shape:
        raise ValueError(
            "mask_zero_torque_history_inputs expects q, qd, and tau_model to share shape; "
            f"got {q_arr.shape}, {qd_arr.shape}, {tau_arr.shape}."
        )
    if ref_tau.shape != tau_arr.shape:
        raise ValueError(
            f"raw_tau shape {ref_tau.shape} does not match tau_model shape {tau_arr.shape}."
        )
    zero_mask = zero_torque_history_keep_mask(ref_tau, threshold=threshold) <= 0.0
    zero_mask = xp.expand_dims(zero_mask, axis=-1)
    q_masked = xp.where(zero_mask, 0.0, q_arr)
    qd_masked = xp.where(zero_mask, 0.0, qd_arr)
    tau_masked = xp.where(zero_mask, 0.0, tau_arr)
    return q_masked, qd_masked, tau_masked


def mask_history_inputs_by_keep_mask(
    q,
    qd,
    tau_model,
    keep_mask,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero invalid rows according to an explicit keep mask."""
    xp = _array_module(q, qd, tau_model, keep_mask)
    q_arr = xp.asarray(q, dtype=xp.float32)
    qd_arr = xp.asarray(qd, dtype=xp.float32)
    tau_arr = xp.asarray(tau_model, dtype=xp.float32)
    keep_arr = xp.asarray(keep_mask, dtype=xp.float32).reshape(-1)
    if q_arr.shape != qd_arr.shape or q_arr.shape != tau_arr.shape:
        raise ValueError(
            "mask_history_inputs_by_keep_mask expects q, qd, and tau_model to share shape; "
            f"got {q_arr.shape}, {qd_arr.shape}, {tau_arr.shape}."
        )
    if q_arr.ndim != 2:
        raise ValueError(
            "mask_history_inputs_by_keep_mask expects [N, DoF] arrays, "
            f"got q shape {q_arr.shape}."
        )
    if keep_arr.shape != (q_arr.shape[0],):
        raise ValueError(
            f"keep_mask must have shape {(q_arr.shape[0],)}, got {keep_arr.shape}."
        )
    invalid = xp.expand_dims(keep_arr <= 0.0, axis=-1)
    q_masked = xp.where(invalid, 0.0, q_arr)
    qd_masked = xp.where(invalid, 0.0, qd_arr)
    tau_masked = xp.where(invalid, 0.0, tau_arr)
    return q_masked, qd_masked, tau_masked


def prepare_history_inputs(
    q,
    qd,
    tau,
    *,
    gravity=None,
    ideal_model_has_gravity: bool,
    context: str,
    tau_is_model_space: bool = False,
    apply_zero_torque_mask: bool = False,
    raw_tau=None,
    threshold: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared deploy-time history input preparation for static and streaming paths.

    Returns:
        q_out, qd_out, tau_model_out, keep_mask
    """
    xp = _array_module(q, qd, tau, gravity, raw_tau)
    q_arr = xp.asarray(q, dtype=xp.float32)
    qd_arr = xp.asarray(qd, dtype=xp.float32)
    raw_tau_arr = xp.asarray(tau, dtype=xp.float32)
    tau_model = prepare_model_space_torque(
        raw_tau_arr,
        gravity,
        ideal_model_has_gravity=ideal_model_has_gravity,
        context=context,
        tau_is_model_space=tau_is_model_space,
    )
    keep = zero_torque_history_keep_mask(
        raw_tau_arr if raw_tau is None else raw_tau,
        threshold=threshold,
    )
    if not apply_zero_torque_mask:
        return q_arr, qd_arr, tau_model, keep
    q_masked, qd_masked, tau_masked = mask_zero_torque_history_inputs(
        q_arr,
        qd_arr,
        tau_model,
        raw_tau=raw_tau_arr if raw_tau is None else raw_tau,
        threshold=threshold,
    )
    return q_masked, qd_masked, tau_masked, keep


def _first_present(sample: dict, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in sample and sample[key] is not None:
            arr = np.asarray(sample[key], dtype=np.float32).reshape(-1)
            if arr.size:
                return arr
    return None


def extract_history_window_arrays(
    window: Sequence[dict],
    *,
    dof: int = 7,
    q_keys: Sequence[str] = ("q", "qpos"),
    dq_keys: Sequence[str] = ("dq", "qd", "qvel"),
    tau_keys: Sequence[str] = ("tau_applied", "tau_cmd", "tau_commanded", "tau", "u_des", "u", "tau_measured"),
    gravity_keys: Sequence[str] = ("gravity",),
    t_keys: Sequence[str] = ("t", "t_raw", "timestamp"),
    valid_keys: Sequence[str] = ("valid_for_history",),
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
]:
    """Convert a controller-published history window into aligned arrays."""
    ts_list = []
    q_list = []
    dq_list = []
    tau_list = []
    gravity_list = []
    keep_list = []
    have_all_gravity = True
    have_any_keep = False

    for sample in window:
        if not isinstance(sample, dict):
            continue
        t = None
        for tk in t_keys:
            if tk in sample and sample[tk] is not None:
                try:
                    t = float(sample[tk])
                    break
                except Exception:
                    t = None
        if t is None:
            continue
        q = _first_present(sample, q_keys)
        dq = _first_present(sample, dq_keys)
        tau = _first_present(sample, tau_keys)
        gravity = _first_present(sample, gravity_keys)
        if q is None or dq is None or tau is None:
            continue
        if q.size < dof or dq.size < dof or tau.size < dof:
            continue
        ts_list.append(t)
        q_list.append(q[:dof])
        dq_list.append(dq[:dof])
        tau_list.append(tau[:dof])
        keep_value = None
        for valid_key in valid_keys:
            if valid_key in sample and sample[valid_key] is not None:
                keep_value = 1.0 if bool(sample[valid_key]) else 0.0
                have_any_keep = True
                break
        keep_list.append(1.0 if keep_value is None else float(keep_value))
        if gravity is None or gravity.size < dof:
            have_all_gravity = False
            gravity_list.append(np.zeros((dof,), dtype=np.float32))
        else:
            gravity_list.append(gravity[:dof])

    if not ts_list:
        return None
    t_arr = np.asarray(ts_list, dtype=np.float64)
    q_arr = np.asarray(q_list, dtype=np.float32)
    dq_arr = np.asarray(dq_list, dtype=np.float32)
    tau_arr = np.asarray(tau_list, dtype=np.float32)
    gravity_arr = np.asarray(gravity_list, dtype=np.float32) if have_all_gravity else None
    keep_arr = np.asarray(keep_list, dtype=np.float32) if have_any_keep else None
    return t_arr, q_arr, dq_arr, tau_arr, gravity_arr, keep_arr


# Backward-compatible aliases used by existing tests/callers.
_mask_zero_torque_history_inputs = mask_zero_torque_history_inputs
_zero_torque_history_keep_mask = zero_torque_history_keep_mask


def apply_history_fusion(
    fusion_params: Any,
    history_emb_applied: Any,
    history_emb_base: Any,
    history_emb_tam: Any,
) -> Any:
    """Linear fusion of the applied/base/TAM history embeddings (``base_tam_fusion`` checkpoints).

    Shared by the workstation mapping server and the offline/sim evaluators so the
    simulator does not need the mapping server's transport dependencies.
    """
    import jax.numpy as jnp

    x = jnp.concatenate(
        [
            jnp.asarray(history_emb_applied, dtype=jnp.float32),
            jnp.asarray(history_emb_base, dtype=jnp.float32),
            jnp.asarray(history_emb_tam, dtype=jnp.float32),
        ],
        axis=-1,
    )
    return jnp.einsum("...i,ij->...j", x, fusion_params["kernel"]) + fusion_params["bias"]

