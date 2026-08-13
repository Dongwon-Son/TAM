#!/usr/bin/env python3
"""Independent online DAgger-like finetuning for TAM.

This trainer starts from an existing TAM checkpoint, rolls out a closed-loop
TAM controller online, then treats that episode as stopped behavior data. It
recomputes compact history-token embeddings from the stopped trace and
supervises only a small number of sampled query points after each token.
Rollout state, labels, and behavior actions are stopped; the supervised
gradient path is:

    teacher/adaptor loss -> adaptor -> sampled history tokens -> history encoder

By default, each query expands into a small desired-torque map using the same
tau-sampling trick as the offline trainer. The history conditioning also keeps
base-policy and TAM-residual histories distinguishable by fusing three stopped
history embeddings: applied torque, base torque, and TAM residual torque.
Behavior rollout uses fixed episode-level teacher-alpha anchors plus the current
TAM residual, with independent pre/post-switch alpha values. If both behavior
alpha endpoints are zero, rollout takes the base-policy anchor directly and
skips teacher-torque computation until supervision. Supervision uses the same
alpha regime unless explicitly overridden.

The trainer intentionally lives outside the original offline TAM trainer so that
the data-generation and online-history assumptions stay explicit.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import struct as flax_struct
from flax.training import checkpoints, train_state
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax
import wandb

from simadaptor.cli import parse_tyro_config
import simadaptor.core.structs as structs
import simadaptor.data.datagen as datagen
from simadaptor.data.datagen_profiles import derive_robot_key, load_datagen_profile
from simadaptor.deploy.inf_util import SimAdaptorInference
from simadaptor.eval.gt_tau_cmd_validation import compute_gt_tau_cmd
from simadaptor.eval.online_runtime import (
    advance_online_history_state,
    align_norm_stats_to_dof,
    apply_online_adaptor,
    build_online_history_runtime,
    init_online_history_state,
    push_window,
    zero_torque_history_keep_mask,
)
import simadaptor.physics.actuator as actuator_util
import simadaptor.physics.rollout as rollout


@dataclasses.dataclass
class OnlineDaggerConfig:
    # Checkpoint resolution. --ckpt is the only checkpoint source.
    ckpt: Optional[str] = None
    checkpoint_step: Optional[int] = None
    xml: Optional[str] = None

    # Datagen profile / episode construction.
    profile_table: str = "assets/datagen_profiles.json"
    profile_key: Optional[str] = None
    episode_duration_s: float = 12.0
    num_waypoints: int = 12
    pause_prob: float = 0.30
    dt: float = 0.001
    switch_mode: str = "token_grid_full"
    loss_stride_steps: Optional[int] = None
    query_samples_per_token: int = 2
    query_window_strides: int = 1
    attention_history_s: Optional[float] = None
    tau_map_sample_no: int = 128
    rollout_cmd_noise_std: Optional[float] = None
    base_correction_alpha_min: float = 0.0
    base_correction_alpha_max: float = 1.0
    history_torque_mode: str = "base_tam_fusion"
    behavior_torque_noise_scale: float = 1.0
    behavior_alpha_min: float = 0.0
    behavior_alpha_max: float = 1.0
    behavior_alpha_mode: str = "binary"
    behavior_alpha_one_prob: float = 0.5
    supervision_alpha_mode: str = "behavior"

    # Hidden force defaults follow the current datagen settings.
    external_force_body_name: str = "hand"
    external_force_use_profile_targets: bool = True
    external_force_num_impulses: int = 5
    external_force_magnitude_min_n: float = 10.0
    external_force_magnitude_max_n: float = 100.0
    external_force_duration_min_s: float = 0.08
    external_force_duration_max_s: float = 0.80

    # Optimization.
    seed: int = 0
    batch_size: int = 2
    max_steps: int = 12_000
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    teacher_loss_weight: float = 1.0
    zero_residual_weight: float = 0.5
    tau_huber_delta: float = 1.0

    # Logging / checkpointing.
    workdir: str = "checkpoints/tam_online_dagger"
    run_name: Optional[str] = None
    wandb_project: str = "tam"
    wandb_mode: str = "disabled"
    wandb_group: Optional[str] = None
    wandb_tags: tuple[str, ...] = ()
    log_interval: int = 10
    ckpt_interval: int = 500
    keep_checkpoints: int = 3
    save_final: bool = True

    # Lightweight periodic rollout-tracking diagnostics.
    exp4_eval_interval: int = 0
    exp4_eval_warmup: bool = False
    exp4_eval_num_tests: int = 2
    exp4_eval_history_s: float = 6.0
    exp4_eval_test_window_s: float = 6.0
    exp4_eval_num_waypoints: int = 12
    exp4_eval_pause_prob: float = 0.30
    exp4_eval_q_noise_std_deg: float = 0.0
    exp4_eval_qd_noise_std_deg_s: float = 0.0
    exp4_eval_external_force_num_impulses: int = 0
    exp4_eval_external_force_min_n: float = 10.0
    exp4_eval_external_force_max_n: float = 100.0
    exp4_eval_external_force_duration_min_s: float = 0.08
    exp4_eval_external_force_duration_max_s: float = 0.80

    # Development checks.
    debug_grad_check: bool = False


class OnlineDaggerState(train_state.TrainState):
    norm_stats: Any = flax_struct.field(pytree_node=True, default=None)


_SAMPLE_RANDOM_PARAM_PROFILE_KEYS = frozenset(
    {
        "armature_min_profile",
        "armature_max_profile",
        "base_kp_profile",
        "white_base_profile",
        "walk_base_profile",
        "white_scale_range",
        "walk_scale_range",
        "kp_scale_small_range",
        "kp_scale_large_range",
        "kd_scale_range",
        "kp_small_prob",
        "ee_payload_mass_delta_range",
        "ee_payload_com_offset_min_local_m",
        "ee_payload_com_offset_max_local_m",
        "joint_model_major_ee_scale",
        "joint_model_major_global_scale",
    }
)


def _require_checkpoint(cfg: OnlineDaggerConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("--ckpt is required; it is the only checkpoint source.")


def _resolve_tam_seq_length(base_cfg: Any) -> int:
    # Prefer raw pickled values: on legacy checkpoint configs the compat
    # properties shadow the instance __dict__ with branch class defaults.
    inst = getattr(base_cfg, "__dict__", None)
    value = None
    if isinstance(inst, dict):
        value = inst.get("tam_seq_length")
        if value is None:
            value = inst.get("adaptor_seq_length")
    if value is None:
        value = getattr(base_cfg, "tam_seq_length", None)
    if value is None:
        value = getattr(base_cfg, "adaptor_seq_length", None)
    return max(int(value or 1), 1)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def _to_float(x: Any) -> float:
    arr = np.asarray(jax.device_get(x))
    return float(arr.reshape(()))


def _fit_to_model_dim(x: jax.Array, target_dim: int) -> jax.Array:
    x = jnp.asarray(x)
    dim = int(x.shape[-1])
    if dim == target_dim:
        return x
    if dim > target_dim:
        return x[..., :target_dim]
    pad_shape = x.shape[:-1] + (target_dim - dim,)
    return jnp.concatenate([x, jnp.zeros(pad_shape, dtype=x.dtype)], axis=-1)


def _fit_vec_or_zero(x: Optional[jax.Array], dim: int, dtype=jnp.float32) -> jax.Array:
    if x is None:
        return jnp.zeros((dim,), dtype=dtype)
    return _fit_to_model_dim(jnp.asarray(x, dtype=dtype), dim)


def _select_rollout_params(
    use_b: jax.Array,
    params_a: structs.RolloutParams,
    params_b: structs.RolloutParams,
) -> structs.RolloutParams:
    def _choose(a, b):
        if a is None:
            return None
        return jnp.where(use_b, b, a)

    return jax.tree.map(_choose, params_a, params_b)


def _select_rollout_params_batched(
    use_b: jax.Array,
    params_a: structs.RolloutParams,
    params_b: structs.RolloutParams,
) -> structs.RolloutParams:
    use_b = jnp.asarray(use_b)

    def _choose(a, b):
        if a is None:
            return None
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        out_shape = use_b.shape + a.shape
        mask = use_b.reshape(use_b.shape + (1,) * a.ndim)
        return jnp.where(mask, jnp.broadcast_to(b, out_shape), jnp.broadcast_to(a, out_shape))

    return jax.tree.map(_choose, params_a, params_b)


def _replace_kp_kd(params: structs.RolloutParams, kp: jax.Array, kd: jax.Array) -> structs.RolloutParams:
    return params.replace(kp=kp, kd=kd)


def _stop_tree(tree):
    return jax.tree.map(jax.lax.stop_gradient, tree)


def _stop_online_state_carry(state):
    """Detach online rollout/history carry for behavior-only closed-loop scans."""
    return state.replace(
        history_emb=jax.lax.stop_gradient(state.history_emb),
        cache=_stop_tree(state.cache),
        q_buf=jax.lax.stop_gradient(state.q_buf),
        qd_buf=jax.lax.stop_gradient(state.qd_buf),
        tau_buf=jax.lax.stop_gradient(state.tau_buf),
        keep_buf=jax.lax.stop_gradient(state.keep_buf),
        sample_count=jax.lax.stop_gradient(state.sample_count),
        next_emit_idx=jax.lax.stop_gradient(state.next_emit_idx),
        has_embedding=jax.lax.stop_gradient(state.has_embedding),
    )


def _finite_or_zero(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))


def _masked_mean(x: jax.Array, mask: jax.Array, eps: float = 1.0e-6) -> jax.Array:
    x = jnp.asarray(x)
    mask = jnp.asarray(mask, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), eps)


def _masked_rms(x: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(_masked_mean(jnp.square(x), mask), 0.0))


def _tau_ref_noise_std(rollout_cmd_noise_std: float, dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(0.5 * float(rollout_cmd_noise_std), dtype=dtype)


def _sample_tau_ref_candidates(
    tau_center: jax.Array,
    tau_noise_key: jax.Array,
    *,
    tau_map_sample_no: int,
    rollout_cmd_noise_std: float,
) -> jax.Array:
    tau_center = jax.lax.stop_gradient(tau_center)
    sample_no = int(tau_map_sample_no)
    if sample_no <= 0:
        raise ValueError(f"tau_map_sample_no must be positive, got {sample_no}.")
    if sample_no == 1:
        return tau_center[:, None, :]
    tau_ref_noise = jax.random.normal(
        tau_noise_key,
        shape=(int(tau_center.shape[0]), sample_no - 1, int(tau_center.shape[-1])),
        dtype=tau_center.dtype,
    ) * _tau_ref_noise_std(rollout_cmd_noise_std, tau_center.dtype)
    return jnp.concatenate(
        [tau_center[:, None, :], tau_center[:, None, :] + tau_ref_noise],
        axis=1,
    )


def _sample_base_correction_alpha(
    key: jax.Array,
    *,
    num_query_samples: int,
    tau_map_sample_no: int,
    alpha_min: float,
    alpha_max: float,
    dtype: jnp.dtype,
) -> jax.Array:
    sample_no = int(tau_map_sample_no)
    alpha = jax.random.uniform(
        key,
        (int(num_query_samples), sample_no),
        minval=float(alpha_min),
        maxval=float(alpha_max),
        dtype=dtype,
    )
    lo = jnp.asarray(float(alpha_min), dtype=dtype)
    hi = jnp.asarray(float(alpha_max), dtype=dtype)
    alpha = alpha.at[:, 0].set(lo)
    if sample_no > 1:
        alpha = alpha.at[:, -1].set(hi)
    return alpha


def _scale_controller_noise(
    controller_params: structs.ControllerParams,
    scale: float,
) -> structs.ControllerParams:
    scale = float(scale)
    if controller_params.torque_noise_std is None or scale == 1.0:
        return controller_params
    return controller_params.replace(
        torque_noise_std=controller_params.torque_noise_std * scale,
    )


def _init_history_fusion_params(emb_dim: int, *, dtype=jnp.float32) -> dict[str, jax.Array]:
    emb_dim = int(emb_dim)
    kernel = jnp.zeros((3 * emb_dim, emb_dim), dtype=dtype)
    kernel = kernel.at[:emb_dim, :].set(jnp.eye(emb_dim, dtype=dtype))
    return {
        "kernel": kernel,
        "bias": jnp.zeros((emb_dim,), dtype=dtype),
    }


def _ensure_history_fusion_params(params: Any, *, emb_dim: int) -> Any:
    if "history_fusion" in params:
        return params
    out = dict(params)
    out["history_fusion"] = _init_history_fusion_params(int(emb_dim))
    return out


def _apply_history_fusion(
    fusion_params: Any,
    history_emb_applied: jax.Array,
    history_emb_base: jax.Array,
    history_emb_tam: jax.Array,
) -> jax.Array:
    x = jnp.concatenate([history_emb_applied, history_emb_base, history_emb_tam], axis=-1)
    return jnp.einsum("...i,ij->...j", x, fusion_params["kernel"]) + fusion_params["bias"]


def _history_embedding_for_mode(
    params: Any,
    history_torque_mode: str,
    history_emb_applied: jax.Array,
    history_emb_base: jax.Array,
    history_emb_tam: jax.Array,
) -> jax.Array:
    mode = str(history_torque_mode)
    if mode == "applied":
        return history_emb_applied
    if mode == "base_tam_fusion":
        return _apply_history_fusion(
            params["history_fusion"],
            history_emb_applied,
            history_emb_base,
            history_emb_tam,
        )
    raise ValueError(f"Unsupported history_torque_mode={history_torque_mode!r}.")


def _resolve_external_force_body(xml_path: Path, body_name: str) -> tuple[int, str]:
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    requested = str(body_name or "").strip()
    if requested and requested.lower() not in ("auto", "ee", "end_effector"):
        body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, requested))
        if body_id < 0:
            raise ValueError(f"Could not resolve external-force body {requested!r} in {xml_path}.")
        return body_id, requested

    if int(mj_model.nsite) > 0:
        body_id = int(np.asarray(mj_model.site_bodyid, dtype=np.int32)[-1])
    else:
        dof_bodyid = np.asarray(mj_model.dof_bodyid, dtype=np.int32)
        if dof_bodyid.size == 0:
            raise ValueError(f"Could not auto-resolve an external-force body for {xml_path}.")
        body_id = int(dof_bodyid[min(max(int(mj_model.nu) - 1, 0), int(dof_bodyid.size) - 1)])
    resolved = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    return body_id, str(resolved)


def _resolve_external_force_targets(
    xml_path: Path,
    *,
    body_name: str,
    profile_kwargs: dict[str, Any],
    use_profile_targets: bool,
) -> tuple[np.ndarray, tuple[str, ...], Optional[np.ndarray], Optional[np.ndarray]]:
    profile_targets = profile_kwargs.get("external_force_targets", None) if use_profile_targets else None
    if profile_targets:
        targets = list(profile_targets)
    else:
        targets = [{"body_name": body_name}]

    body_ids: list[int] = []
    body_names: list[str] = []
    pos_mins: list[np.ndarray] = []
    pos_maxs: list[np.ndarray] = []
    has_any_position_box = False
    has_all_position_boxes = True
    for target in targets:
        resolved_id, resolved_name = _resolve_external_force_body(xml_path, str(target["body_name"]))
        body_ids.append(int(resolved_id))
        body_names.append(str(resolved_name))
        pos_min = target.get("position_min_local_m", None)
        pos_max = target.get("position_max_local_m", None)
        has_box = pos_min is not None and pos_max is not None
        has_any_position_box = has_any_position_box or has_box
        has_all_position_boxes = has_all_position_boxes and has_box
        if has_box:
            pos_mins.append(np.asarray(pos_min, dtype=np.float32).reshape(3))
            pos_maxs.append(np.asarray(pos_max, dtype=np.float32).reshape(3))

    if has_any_position_box and not has_all_position_boxes:
        raise ValueError(
            "External-force targets must either all define local position boxes or none do."
        )
    body_ids_np = np.asarray(body_ids, dtype=np.int32)
    if has_all_position_boxes:
        return (
            body_ids_np,
            tuple(body_names),
            np.stack(pos_mins, axis=0).astype(np.float32),
            np.stack(pos_maxs, axis=0).astype(np.float32),
        )
    return body_ids_np, tuple(body_names), None, None


def _build_optimizer(cfg: OnlineDaggerConfig) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if cfg.grad_clip_norm and cfg.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(float(cfg.grad_clip_norm)))
    transforms.append(optax.adamw(float(cfg.lr), weight_decay=float(cfg.weight_decay)))
    return optax.chain(*transforms)


def _copy_robot_assets(source_xml: Path, out_dir: Path) -> None:
    robot_dir = out_dir / "robot_model"
    robot_dir.mkdir(parents=True, exist_ok=True)
    if source_xml.exists():
        dst = robot_dir / "robot.xml"
        if source_xml.resolve() != dst.resolve():
            shutil.copy2(source_xml, dst)
        selfcol = source_xml.with_name(f"{source_xml.stem}_selfcol.xml")
        if selfcol.exists():
            shutil.copy2(selfcol, robot_dir / "robot_selfcol.xml")


def _write_metadata(
    out_dir: Path,
    *,
    base_cfg: Any,
    dagger_cfg: OnlineDaggerConfig,
    params: Any,
    norm_stats: Any,
    source_checkpoint: Optional[Path],
    xml_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_robot_assets(xml_path, out_dir)
    with (out_dir / "save_dict.pkl").open("wb") as f:
        pickle.dump(
            {
                "cfg": base_cfg,
                "dagger_cfg": dagger_cfg,
                "params": params,
                "norm_stats": norm_stats,
                # Record the checkpoint reference as the user provided it; a
                # resolved absolute path would embed the local machine layout
                # in every published checkpoint.
                "source_checkpoint": str(dagger_cfg.ckpt) if dagger_cfg.ckpt else None,
            },
            f,
        )
    with (out_dir / "online_dagger_config.json").open("w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(dagger_cfg), f, indent=2, default=_json_default)


def _save_checkpoint(
    out_dir: Path,
    state: OnlineDaggerState,
    *,
    step: int,
    cfg: OnlineDaggerConfig,
    base_cfg: Any,
    source_checkpoint: Optional[Path],
    xml_path: Path,
) -> None:
    checkpoints.save_checkpoint(
        ckpt_dir=str(out_dir),
        target=state,
        step=int(step),
        overwrite=True,
        keep=int(cfg.keep_checkpoints),
    )
    _write_metadata(
        out_dir,
        base_cfg=base_cfg,
        dagger_cfg=cfg,
        params=state.params,
        norm_stats=state.norm_stats,
        source_checkpoint=source_checkpoint,
        xml_path=xml_path,
    )


def build_loss(
    *,
    cfg: OnlineDaggerConfig,
    base_cfg: Any,
    hist_model: Any,
    adaptor_model: Any,
    mjx_model: mjx.Model,
    runtime: Any,
    norm_stats: Any,
    profile_kwargs: dict[str, Any],
    external_force_body_ids_np: np.ndarray,
    external_force_position_min_np: Optional[np.ndarray],
    external_force_position_max_np: Optional[np.ndarray],
    arm_ids_np: np.ndarray,
) -> Any:
    arm_ids = jnp.asarray(arm_ids_np, dtype=jnp.int32)
    dof = int(arm_ids_np.shape[0])
    total_steps = int(round(float(cfg.episode_duration_s) / float(cfg.dt)))
    if total_steps < 2:
        raise ValueError(f"episode_duration_s/dt must produce at least 2 steps, got {total_steps}.")

    adaptor_seq_length = _resolve_tam_seq_length(base_cfg)
    patch_size = int(runtime.config.patch_size)
    patch_stride = int(runtime.config.patch_stride)
    context_half = int(runtime.config.context_half)
    grid_stride = int(cfg.loss_stride_steps or patch_stride)
    if grid_stride <= 0:
        raise ValueError(f"loss_stride_steps must be positive, got {grid_stride}.")
    query_samples_per_token = int(cfg.query_samples_per_token)
    if query_samples_per_token <= 0:
        raise ValueError(
            f"query_samples_per_token must be positive, got {query_samples_per_token}."
        )
    query_window_strides = int(cfg.query_window_strides)
    if query_window_strides <= 0:
        raise ValueError(f"query_window_strides must be positive, got {query_window_strides}.")
    query_window_steps = max(int(grid_stride) * query_window_strides, 1)
    if cfg.attention_history_s is not None and float(cfg.attention_history_s) < 0.0:
        raise ValueError(
            "attention_history_s must be non-negative when set; "
            f"got {cfg.attention_history_s}."
        )
    tau_map_sample_no = int(cfg.tau_map_sample_no)
    if tau_map_sample_no <= 0:
        raise ValueError(f"tau_map_sample_no must be positive, got {tau_map_sample_no}.")
    history_torque_mode = str(cfg.history_torque_mode)
    if history_torque_mode not in ("applied", "base_tam_fusion"):
        raise ValueError(
            "history_torque_mode must be one of {'applied', 'base_tam_fusion'}, "
            f"got {history_torque_mode!r}."
        )
    supervision_alpha_mode = str(cfg.supervision_alpha_mode)
    if supervision_alpha_mode not in ("sampled", "behavior"):
        raise ValueError(
            "supervision_alpha_mode must be one of {'sampled', 'behavior'}, "
            f"got {supervision_alpha_mode!r}."
        )
    behavior_noise_scale = float(cfg.behavior_torque_noise_scale)
    if not np.isfinite(behavior_noise_scale) or behavior_noise_scale < 0.0:
        raise ValueError(
            "behavior_torque_noise_scale must be finite and non-negative, "
            f"got {behavior_noise_scale}."
        )
    behavior_alpha_min = float(cfg.behavior_alpha_min)
    behavior_alpha_max = float(cfg.behavior_alpha_max)
    if not (0.0 <= behavior_alpha_min <= behavior_alpha_max <= 1.0):
        raise ValueError(
            "behavior_alpha_min/max must satisfy 0 <= min <= max <= 1, "
            f"got min={behavior_alpha_min}, max={behavior_alpha_max}."
        )
    behavior_alpha_mode = str(cfg.behavior_alpha_mode)
    if behavior_alpha_mode not in ("binary", "uniform"):
        raise ValueError(
            "behavior_alpha_mode must be one of {'binary', 'uniform'}, "
            f"got {behavior_alpha_mode!r}."
        )
    behavior_alpha_one_prob = float(cfg.behavior_alpha_one_prob)
    if not (0.0 <= behavior_alpha_one_prob <= 1.0):
        raise ValueError(
            "behavior_alpha_one_prob must satisfy 0 <= p <= 1, "
            f"got {behavior_alpha_one_prob}."
        )
    behavior_uses_teacher_anchor = not (
        behavior_alpha_min == 0.0 and behavior_alpha_max == 0.0
    )
    alpha_min = float(cfg.base_correction_alpha_min)
    alpha_max = float(cfg.base_correction_alpha_max)
    if not (0.0 <= alpha_min <= alpha_max <= 1.0):
        raise ValueError(
            "base_correction_alpha_min/max must satisfy 0 <= min <= max <= 1, "
            f"got min={alpha_min}, max={alpha_max}."
        )

    if cfg.switch_mode != "token_grid_full":
        raise ValueError(f"Unsupported switch_mode={cfg.switch_mode!r}; only 'token_grid_full' is implemented.")
    first_switch = patch_size + context_half
    last_switch = total_steps - grid_stride
    if last_switch < 1:
        last_switch = total_steps - 1
    first_switch = min(max(first_switch, 1), last_switch)
    n_switch_choices = max(1, (last_switch - first_switch) // grid_stride + 1)

    external_force_body_ids_np = np.asarray(external_force_body_ids_np, dtype=np.int32).reshape((-1,))
    if int(external_force_body_ids_np.shape[0]) <= 0:
        external_force_body_ids_np = np.asarray([-1], dtype=np.int32)
    external_force_body_ids = jnp.asarray(external_force_body_ids_np, dtype=jnp.int32)
    num_force_targets = int(external_force_body_ids_np.shape[0])
    force_has_position_box = (
        external_force_position_min_np is not None
        and external_force_position_max_np is not None
    )
    if force_has_position_box:
        external_force_position_min = jnp.asarray(external_force_position_min_np, dtype=jnp.float32)
        external_force_position_max = jnp.asarray(external_force_position_max_np, dtype=jnp.float32)
        if external_force_position_min.shape != (num_force_targets, 3):
            raise ValueError(
                "external_force_position_min_np must have shape "
                f"({num_force_targets}, 3), got {external_force_position_min.shape}."
            )
        if external_force_position_max.shape != (num_force_targets, 3):
            raise ValueError(
                "external_force_position_max_np must have shape "
                f"({num_force_targets}, 3), got {external_force_position_max.shape}."
            )
    else:
        external_force_position_min = jnp.zeros((num_force_targets, 3), dtype=jnp.float32)
        external_force_position_max = jnp.zeros((num_force_targets, 3), dtype=jnp.float32)
    hidden_force_enabled = bool(
        cfg.external_force_num_impulses > 0
        and num_force_targets > 0
        and np.any(external_force_body_ids_np >= 0)
    )
    norm_stats_aligned = align_norm_stats_to_dof(norm_stats, dof)
    sample_random_param_kwargs = {
        k: v for k, v in profile_kwargs.items() if k in _SAMPLE_RANDOM_PARAM_PROFILE_KEYS
    }
    waypoint_max_delta_deg_profile = profile_kwargs.get("waypoint_max_delta_deg_profile", None)
    external_force_magnitude_min_n = float(
        profile_kwargs.get("external_force_magnitude_min_n", cfg.external_force_magnitude_min_n)
    )
    external_force_magnitude_max_n = float(
        profile_kwargs.get("external_force_magnitude_max_n", cfg.external_force_magnitude_max_n)
    )
    rollout_cmd_noise_std = (
        float(profile_kwargs.get("rollout_cmd_noise_std", 0.0))
        if cfg.rollout_cmd_noise_std is None
        else float(cfg.rollout_cmd_noise_std)
    )
    if not np.isfinite(rollout_cmd_noise_std):
        raise ValueError(f"rollout_cmd_noise_std must be finite, got {rollout_cmd_noise_std}.")

    actuator_trnid = jnp.asarray(mjx_model.actuator_trnid, dtype=jnp.int32)
    jnt_dofadr = jnp.asarray(mjx_model.jnt_dofadr, dtype=jnp.int32)
    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = jnp.clip(act_jnt_id, 0, int(mjx_model.njnt) - 1)
    actuator_dof_abs = jnp.where(
        (act_jnt_id >= 0) & (act_jnt_id < int(mjx_model.njnt)),
        jnt_dofadr[act_jnt_id_clamped],
        jnp.minimum(jnp.arange(int(mjx_model.nu), dtype=jnp.int32), int(mjx_model.nv) - 1),
    )
    joint_range = jnp.asarray(mjx_model.jnt_range, dtype=jnp.float32)[arm_ids]
    data_template = mjx.make_data(mjx_model)
    num_force_impulses = max(int(cfg.external_force_num_impulses), 0)

    def compute_external_wrench(
        data: mjx.Data,
        rollout_params: structs.RolloutParams,
        force_terms_t: jax.Array,
        force_local_positions: jax.Array,
        force_body_id: jax.Array,
    ) -> jax.Array:
        force_terms_t = jnp.asarray(force_terms_t, dtype=data.qpos.dtype)
        if force_has_position_box:
            body_id = jnp.asarray(force_body_id, dtype=jnp.int32)
            body_id_clamped = jnp.clip(body_id, 0, int(mjx_model.nbody) - 1)
            model_step = rollout_params.set_mjx_model(mjx_model)
            state_for_force = mjx.forward(
                model_step,
                data.replace(xfrc_applied=jnp.zeros_like(data.xfrc_applied)),
            )
            body_rot = jnp.asarray(state_for_force.xmat[body_id_clamped], dtype=data.qpos.dtype)
            if body_rot.ndim == 1:
                body_rot = body_rot.reshape((3, 3))
            body_ipos_local = (
                jnp.asarray(rollout_params.body_ipos[body_id_clamped], dtype=data.qpos.dtype)
                if rollout_params.body_ipos is not None
                else jnp.asarray(mjx_model.body_ipos[body_id_clamped], dtype=data.qpos.dtype)
            )
            lever_local = jnp.asarray(force_local_positions, dtype=data.qpos.dtype) - body_ipos_local[None, :]
            lever_world = jnp.einsum("ij,kj->ki", body_rot, lever_local)
            torque_terms_t = jnp.cross(lever_world, force_terms_t, axis=-1)
            return jnp.concatenate(
                [jnp.sum(force_terms_t, axis=0), jnp.sum(torque_terms_t, axis=0)],
                axis=-1,
            )
        return jnp.sum(force_terms_t, axis=0)

    def step_dynamics(
        data: mjx.Data,
        tau_cmd: jax.Array,
        rollout_params: structs.RolloutParams,
        external_force_t: jax.Array,
        external_force_body_id_t: jax.Array,
    ) -> mjx.Data:
        qpos_for_act = data.qpos[actuator_dof_abs]
        qvel_for_act = data.qvel[actuator_dof_abs]
        tau_eff = actuator_util.actuator_model(
            tau_cmd,
            qpos_for_act,
            qvel_for_act,
            rollout_params.actuator_params,
        )
        ctrl_full = jnp.zeros((int(mjx_model.nu),), dtype=tau_eff.dtype).at[: tau_eff.shape[0]].set(tau_eff)
        model_step = rollout_params.set_mjx_model(mjx_model)
        xfrc_applied = jnp.zeros_like(data.xfrc_applied)
        if hidden_force_enabled:
            body_id = jnp.asarray(external_force_body_id_t, dtype=jnp.int32)
            body_id_clamped = jnp.clip(body_id, 0, int(mjx_model.nbody) - 1)
            force_t = jnp.asarray(external_force_t, dtype=xfrc_applied.dtype)
            if force_has_position_box:
                xfrc_candidate = xfrc_applied.at[body_id_clamped, :].set(force_t[:6])
            else:
                xfrc_candidate = xfrc_applied.at[body_id_clamped, :3].set(force_t[:3])
            xfrc_applied = jax.lax.cond(
                (body_id >= 0) & (body_id < int(mjx_model.nbody)),
                lambda _: xfrc_candidate,
                lambda _: xfrc_applied,
                operand=None,
            )
        return mjx.step(model_step, data.replace(ctrl=ctrl_full, xfrc_applied=xfrc_applied))

    def sample_params(key: jax.Array) -> tuple[structs.RolloutParams, structs.RolloutParams]:
        _, pert_a_dict, _ = datagen.sample_random_params(
            key,
            mjx_model,
            evaluation_mode=False,
            **sample_random_param_kwargs,
        )
        _, pert_b_dict, _ = datagen.sample_random_params(
            jax.random.fold_in(key, 1),
            mjx_model,
            evaluation_mode=False,
            **sample_random_param_kwargs,
        )
        params_a = structs.RolloutParams(**pert_a_dict).fit_model_size(mjx_model)
        params_b = structs.RolloutParams(**pert_b_dict).fit_model_size(mjx_model)
        params_b = _replace_kp_kd(params_b, params_a.kp, params_a.kd)
        return params_a, params_b

    def make_episode_inputs(key: jax.Array, params_a: structs.RolloutParams) -> tuple[jax.Array, ...]:
        k_wps, k_force, k_force_pos, k_force_target, k_qn, k_dqn, k_ctrl, k_switch = jax.random.split(key, 8)
        waypoints = rollout.generate_waypoints(
            k_wps,
            int(cfg.num_waypoints),
            batch_n=1,
            dof=dof,
            joint_range=joint_range,
            pause_prob=float(cfg.pause_prob),
            waypoint_max_delta_deg_profile=waypoint_max_delta_deg_profile,
        )[0]
        q_ref, qd_ref = rollout.build_traj_from_waypoints(
            waypoints,
            total_steps,
            float(cfg.episode_duration_s),
        )
        if hidden_force_enabled:
            force_impulse_terms = rollout._sample_external_force_impulse_components(
                k_force,
                batch_n=1,
                total_steps=total_steps,
                dt=float(cfg.dt),
                num_impulses=num_force_impulses,
                magnitude_min_n=external_force_magnitude_min_n,
                magnitude_max_n=external_force_magnitude_max_n,
                duration_min_s=float(cfg.external_force_duration_min_s),
                duration_max_s=float(cfg.external_force_duration_max_s),
                dtype=jnp.float32,
            )[0]
            force_impulse_terms = jnp.swapaxes(force_impulse_terms, 0, 1)
            force_target_idx = jax.random.randint(k_force_target, (), 0, num_force_targets)
            force_body_id = external_force_body_ids[force_target_idx]
            if force_has_position_box:
                force_local_positions = rollout.sample_external_force_local_positions(
                    k_force_pos,
                    batch_n=1,
                    num_impulses=num_force_impulses,
                    position_min_local_m=external_force_position_min[force_target_idx],
                    position_max_local_m=external_force_position_max[force_target_idx],
                    dtype=jnp.float32,
                )[0]
            else:
                force_local_positions = jnp.zeros((num_force_impulses, 3), dtype=jnp.float32)
        else:
            force_impulse_terms = jnp.zeros((total_steps, num_force_impulses, 3), dtype=jnp.float32)
            force_local_positions = jnp.zeros((num_force_impulses, 3), dtype=jnp.float32)
            force_body_id = jnp.asarray(-1, dtype=jnp.int32)
        q_noise_unit = jax.random.normal(k_qn, (total_steps, dof), dtype=jnp.float32)
        dq_noise_unit = jax.random.normal(k_dqn, (total_steps, dof), dtype=jnp.float32)
        ctrl_keys = jax.random.split(k_ctrl, total_steps)
        switch_choice = jax.random.randint(k_switch, (), 0, n_switch_choices)
        switch_idx = jnp.asarray(first_switch + switch_choice * grid_stride, dtype=jnp.int32)
        if params_a.q_noise_std is None:
            q_noise_unit = jnp.zeros_like(q_noise_unit)
        if params_a.dq_noise_std is None:
            dq_noise_unit = jnp.zeros_like(dq_noise_unit)
        return (
            q_ref,
            qd_ref,
            force_impulse_terms,
            force_local_positions,
            force_body_id,
            q_noise_unit,
            dq_noise_unit,
            ctrl_keys,
            switch_idx,
        )

    def single_episode_loss(params: Any, key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        key_params, key_inputs, key_query, key_tau, key_alpha, key_behavior_alpha = jax.random.split(key, 6)
        params_a, params_b = sample_params(key_params)
        (
            q_ref,
            qd_ref,
            force_impulse_terms,
            force_local_positions,
            force_body_id,
            q_noise_unit,
            dq_noise_unit,
            ctrl_keys,
            switch_idx,
        ) = make_episode_inputs(
            key_inputs,
            params_a,
        )
        if behavior_alpha_mode == "binary":
            alpha_bits = jax.random.bernoulli(key_behavior_alpha, behavior_alpha_one_prob, (2,))
            behavior_alpha_pair = jnp.where(
                alpha_bits,
                jnp.asarray(behavior_alpha_max, dtype=jnp.float32),
                jnp.asarray(behavior_alpha_min, dtype=jnp.float32),
            )
        else:
            behavior_alpha_pair = jax.random.uniform(
                key_behavior_alpha,
                (2,),
                minval=behavior_alpha_min,
                maxval=behavior_alpha_max,
                dtype=jnp.float32,
            )

        data0 = data_template.replace(
            qpos=jnp.zeros((int(mjx_model.nq),), dtype=jnp.float32).at[arm_ids].set(q_ref[0]),
            qvel=jnp.zeros((int(mjx_model.nv),), dtype=jnp.float32).at[arm_ids].set(qd_ref[0]),
            ctrl=jnp.zeros((int(mjx_model.nu),), dtype=jnp.float32),
            xfrc_applied=jnp.zeros_like(data_template.xfrc_applied),
        )
        q_win0 = jnp.repeat(q_ref[0][None, :], adaptor_seq_length, axis=0)
        qd_win0 = jnp.repeat(qd_ref[0][None, :], adaptor_seq_length, axis=0)
        tau_win0 = jnp.zeros((adaptor_seq_length, dof), dtype=jnp.float32)
        ctrl_carry0 = jnp.zeros((1, dof), dtype=jnp.float32)
        online_state_applied0 = init_online_history_state(runtime, dtype=jnp.float32)
        online_state_base0 = init_online_history_state(runtime, dtype=jnp.float32)
        online_state_tam0 = init_online_history_state(runtime, dtype=jnp.float32)

        controller_params = _scale_controller_noise(params_a.controller_params, behavior_noise_scale)
        controller_fn = controller_params.get_actuator_fn(
            control_type="qref",
            ideal_mjx_model=mjx_model,
            add_noise=behavior_noise_scale > 0.0,
        )

        total_steps_f = jnp.asarray(float(total_steps), dtype=jnp.float32)
        dof_f = jnp.asarray(float(dof), dtype=jnp.float32)

        def scan_step(carry, inp):
            data, q_win, qd_win, tau_win, ctrl_carry, online_state_applied, online_state_base, online_state_tam = carry
            abs_idx, q_ref_t, qd_ref_t, force_t, q_noise_u, dq_noise_u, ctrl_key = inp

            use_b = abs_idx >= switch_idx
            active_params = _select_rollout_params(use_b, params_a, params_b)

            q_cur = jax.lax.stop_gradient(data.qpos[arm_ids])
            qd_cur = jax.lax.stop_gradient(data.qvel[arm_ids])
            q_std = _fit_vec_or_zero(active_params.q_noise_std, dof, dtype=q_cur.dtype)
            dq_std = _fit_vec_or_zero(active_params.dq_noise_std, dof, dtype=qd_cur.dtype)
            q_obs = jax.lax.stop_gradient(q_cur + q_noise_u * q_std)
            qd_obs = jax.lax.stop_gradient(qd_cur + dq_noise_u * dq_std)

            q_win = push_window(q_win, q_obs)
            qd_win = push_window(qd_win, qd_obs)
            force_wrench_t = jax.lax.stop_gradient(
                compute_external_wrench(
                    data,
                    active_params,
                    force_t,
                    force_local_positions,
                    force_body_id,
                )
            )

            tau_base_b, _, ctrl_carry = controller_fn(
                q_win[None, ...],
                qd_win[None, ...],
                q_ref_t[None, :],
                qd_ref_t[None, :],
                ctrl_key,
                ctrl_carry,
                u_ref=None,
            )
            tau_base_noisy = jax.lax.stop_gradient(tau_base_b[0])
            tau_win_base = push_window(tau_win, tau_base_noisy)

            has_z = online_state_applied.has_embedding
            behavior_params = _stop_tree(params)
            behavior_history_emb = _history_embedding_for_mode(
                behavior_params,
                history_torque_mode,
                jax.lax.stop_gradient(online_state_applied.history_emb),
                jax.lax.stop_gradient(online_state_base.history_emb),
                jax.lax.stop_gradient(online_state_tam.history_emb),
            )
            if behavior_uses_teacher_anchor:
                teacher_tau_behavior = compute_gt_tau_cmd(
                    mjx_model,
                    active_params,
                    q_obs,
                    qd_obs,
                    tau_base_noisy,
                    external_force_ee=force_wrench_t,
                    external_force_body_id=force_body_id,
                    method="shared_linear",
                )
                behavior_alpha_t = jnp.where(use_b, behavior_alpha_pair[1], behavior_alpha_pair[0])
                behavior_alpha_t = jnp.asarray(behavior_alpha_t, dtype=tau_base_noisy.dtype)
                tau_behavior_anchor = tau_base_noisy + behavior_alpha_t * (
                    jax.lax.stop_gradient(teacher_tau_behavior) - tau_base_noisy
                )
            else:
                behavior_alpha_t = jnp.zeros((), dtype=tau_base_noisy.dtype)
                tau_behavior_anchor = tau_base_noisy
            tau_win_behavior = push_window(tau_win, tau_behavior_anchor)
            _, tam_residual_candidate = apply_online_adaptor(
                adaptor_model=adaptor_model,
                adaptor_apply_fn=None,
                params_adaptor=behavior_params["adaptor"],
                q_window=jax.lax.stop_gradient(q_win),
                qd_window=jax.lax.stop_gradient(qd_win),
                tau_window=jax.lax.stop_gradient(tau_win_behavior),
                history_emb=jax.lax.stop_gradient(behavior_history_emb),
                norm_stats=norm_stats_aligned,
            )
            tam_residual = jnp.where(has_z, tam_residual_candidate, jnp.zeros_like(tau_base_noisy))
            tau_cmd_applied = tau_behavior_anchor + tam_residual
            tau_cmd_applied = jax.lax.stop_gradient(tau_cmd_applied)
            behavior_delta = jax.lax.stop_gradient(tau_cmd_applied - tau_base_noisy)
            tam_residual = jax.lax.stop_gradient(tam_residual)

            data_next = step_dynamics(data, tau_cmd_applied, active_params, force_wrench_t, force_body_id)
            data_next = jax.tree.map(jax.lax.stop_gradient, data_next)
            tau_win_next = push_window(tau_win, tau_cmd_applied)
            online_state_applied_next = advance_online_history_state(
                runtime,
                online_state_applied,
                q_arm=jax.lax.stop_gradient(q_obs),
                qd_arm=jax.lax.stop_gradient(qd_obs),
                tau_arm=jax.lax.stop_gradient(tau_cmd_applied),
                raw_tau_arm=jax.lax.stop_gradient(tau_cmd_applied),
                params_hist=behavior_params["hist"],
                norm_stats=norm_stats_aligned,
            )
            online_state_base_next = advance_online_history_state(
                runtime,
                online_state_base,
                q_arm=jax.lax.stop_gradient(q_obs),
                qd_arm=jax.lax.stop_gradient(qd_obs),
                tau_arm=jax.lax.stop_gradient(tau_base_noisy),
                raw_tau_arm=jax.lax.stop_gradient(tau_cmd_applied),
                params_hist=behavior_params["hist"],
                norm_stats=norm_stats_aligned,
            )
            online_state_tam_next = advance_online_history_state(
                runtime,
                online_state_tam,
                q_arm=jax.lax.stop_gradient(q_obs),
                qd_arm=jax.lax.stop_gradient(qd_obs),
                tau_arm=jax.lax.stop_gradient(tam_residual),
                raw_tau_arm=jax.lax.stop_gradient(tau_cmd_applied),
                params_hist=behavior_params["hist"],
                norm_stats=norm_stats_aligned,
            )
            trace_t = {
                "q_cur": jax.lax.stop_gradient(q_cur),
                "q_ref": jax.lax.stop_gradient(q_ref_t),
                "q_obs": jax.lax.stop_gradient(q_obs),
                "qd_obs": jax.lax.stop_gradient(qd_obs),
                "tau_base_noisy": jax.lax.stop_gradient(tau_base_noisy),
                "tau_cmd_applied": jax.lax.stop_gradient(tau_cmd_applied),
                "behavior_delta": jax.lax.stop_gradient(behavior_delta),
                "tam_residual": jax.lax.stop_gradient(tam_residual),
                "force": force_wrench_t,
                "force_body_id": jax.lax.stop_gradient(force_body_id),
                "q_window": jax.lax.stop_gradient(q_win),
                "qd_window": jax.lax.stop_gradient(qd_win),
                "tau_window_base": jax.lax.stop_gradient(tau_win_base),
                "has_embedding": has_z.astype(jnp.float32),
            }

            return (
                data_next,
                jax.lax.stop_gradient(q_win),
                jax.lax.stop_gradient(qd_win),
                jax.lax.stop_gradient(tau_win_next),
                jax.lax.stop_gradient(ctrl_carry),
                _stop_online_state_carry(online_state_applied_next),
                _stop_online_state_carry(online_state_base_next),
                _stop_online_state_carry(online_state_tam_next),
            ), trace_t

        absidx = jnp.arange(total_steps, dtype=jnp.int32)
        _, trace = jax.lax.scan(
            scan_step,
            (
                data0,
                q_win0,
                qd_win0,
                tau_win0,
                ctrl_carry0,
                online_state_applied0,
                online_state_base0,
                online_state_tam0,
            ),
            (absidx, q_ref, qd_ref, force_impulse_terms, q_noise_unit, dq_noise_unit, ctrl_keys),
        )
        trace = jax.tree.map(jax.lax.stop_gradient, trace)

        hist_q = trace["q_obs"][None, ...]
        hist_qd = trace["qd_obs"][None, ...]
        hist_keep = zero_torque_history_keep_mask(trace["tau_cmd_applied"])[None, ...]

        history_train_window = runtime.config.attention_history_tokens

        def encode_history_tokens(hist_tau: jax.Array) -> jax.Array:
            apply_kwargs = {}
            if history_train_window is not None:
                apply_kwargs["train_window_override"] = int(history_train_window)
            history_emb_all = hist_model.apply(
                params["hist"],
                hist_q,
                hist_qd,
                hist_tau[None, ...],
                deterministic=True,
                norm_stats=norm_stats_aligned,
                input_keep_mask=hist_keep,
                **apply_kwargs,
            )
            if history_emb_all.ndim == 2:
                return history_emb_all[:, None, :][0]
            return history_emb_all[0]

        history_emb_applied_tokens = encode_history_tokens(trace["tau_cmd_applied"])
        if history_torque_mode == "base_tam_fusion":
            history_emb_base_tokens = encode_history_tokens(trace["tau_base_noisy"])
            history_emb_tam_tokens = encode_history_tokens(trace["tam_residual"])
            history_emb_tokens = _history_embedding_for_mode(
                params,
                history_torque_mode,
                history_emb_applied_tokens,
                history_emb_base_tokens,
                history_emb_tam_tokens,
            )
        else:
            history_emb_base_tokens = jnp.zeros_like(history_emb_applied_tokens)
            history_emb_tam_tokens = jnp.zeros_like(history_emb_applied_tokens)
            history_emb_tokens = history_emb_applied_tokens
        n_tokens = int(history_emb_tokens.shape[0])

        token_starts = jnp.arange(n_tokens, dtype=jnp.int32) * int(patch_stride)
        query_base = token_starts + int(patch_size) + int(context_half)
        query_offsets = jax.random.randint(
            key_query,
            (n_tokens, query_samples_per_token),
            minval=0,
            maxval=query_window_steps,
            dtype=jnp.int32,
        )
        query_idx_raw = query_base[:, None] + query_offsets
        query_valid = query_idx_raw < total_steps
        query_idx = jnp.clip(query_idx_raw, 0, total_steps - 1)
        query_idx_flat = query_idx.reshape(-1)
        query_idx_raw_flat = query_idx_raw.reshape(-1)
        query_valid_flat = query_valid.reshape(-1).astype(jnp.float32)
        num_query_samples = int(n_tokens * query_samples_per_token)

        def gather_trace(name: str) -> jax.Array:
            x = trace[name][query_idx]
            return x.reshape((num_query_samples,) + x.shape[2:])

        q_window_flat = gather_trace("q_window")
        qd_window_flat = gather_trace("qd_window")
        tau_window_base_flat = gather_trace("tau_window_base")
        q_obs_flat = gather_trace("q_obs")
        qd_obs_flat = gather_trace("qd_obs")
        tau_base_flat = gather_trace("tau_base_noisy")
        force_flat = gather_trace("force")
        force_body_id_flat = gather_trace("force_body_id")
        behavior_delta_flat = gather_trace("behavior_delta")
        tam_residual_flat = gather_trace("tam_residual")
        history_emb_flat = jnp.repeat(
            history_emb_tokens[:, None, ...],
            query_samples_per_token,
            axis=1,
        ).reshape((num_query_samples,) + history_emb_tokens.shape[1:])

        use_b_flat = query_idx_raw_flat >= switch_idx
        active_params_flat = _select_rollout_params_batched(use_b_flat, params_a, params_b)
        tau_ref_samples = _sample_tau_ref_candidates(
            tau_base_flat,
            key_tau,
            tau_map_sample_no=tau_map_sample_no,
            rollout_cmd_noise_std=rollout_cmd_noise_std,
        )
        teacher_tau_samples = compute_gt_tau_cmd(
            mjx_model,
            active_params_flat,
            q_obs_flat,
            qd_obs_flat,
            tau_ref_samples,
            external_force_ee=force_flat,
            external_force_body_id=force_body_id_flat,
            method="shared_linear",
        )
        teacher_tau_samples = jax.lax.stop_gradient(teacher_tau_samples)
        if supervision_alpha_mode == "behavior":
            correction_alpha_per_query = jnp.where(use_b_flat, behavior_alpha_pair[1], behavior_alpha_pair[0])
            correction_alpha = jnp.broadcast_to(
                correction_alpha_per_query[:, None].astype(tau_ref_samples.dtype),
                (num_query_samples, tau_map_sample_no),
            )
        else:
            correction_alpha = _sample_base_correction_alpha(
                key_alpha,
                num_query_samples=num_query_samples,
                tau_map_sample_no=tau_map_sample_no,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                dtype=tau_ref_samples.dtype,
            )
        tau_base_aug_samples = jax.lax.stop_gradient(
            tau_ref_samples + correction_alpha[..., None] * (teacher_tau_samples - tau_ref_samples)
        )
        target_delta_samples = jax.lax.stop_gradient(teacher_tau_samples - tau_base_aug_samples)

        delta_pred_samples, _ = adaptor_model.apply(
            params["adaptor"],
            q_window_flat,
            qd_window_flat,
            tau_window_base_flat,
            history_emb_flat,
            tau_des_override=tau_base_aug_samples,
            norm_stats=norm_stats_aligned,
        )
        if delta_pred_samples.ndim == 2:
            delta_pred_samples = delta_pred_samples[:, None, :]
        tau_pred_samples = tau_base_aug_samples + delta_pred_samples
        endpoint_delta_pred = delta_pred_samples[:, -1, :]
        endpoint_delta_target = target_delta_samples[:, -1, :]

        teacher_per_sample_tau = jnp.mean(
            optax.huber_loss(tau_pred_samples, teacher_tau_samples, delta=float(cfg.tau_huber_delta)),
            axis=-1,
        )
        teacher_per_sample = jnp.mean(teacher_per_sample_tau, axis=-1)
        endpoint_per_sample = jnp.mean(
            optax.huber_loss(
                endpoint_delta_pred,
                endpoint_delta_target,
                delta=float(cfg.tau_huber_delta),
            ),
            axis=-1,
        )
        loss_per_sample = (
            float(cfg.teacher_loss_weight) * teacher_per_sample
            + float(cfg.zero_residual_weight) * endpoint_per_sample
        )
        tau_err = tau_pred_samples - teacher_tau_samples
        endpoint_delta_err = endpoint_delta_pred - endpoint_delta_target
        total_loss = _masked_mean(loss_per_sample, query_valid_flat)
        supervised_count = jnp.sum(query_valid_flat)
        pre_mask = query_valid_flat * (query_idx_raw_flat < switch_idx).astype(jnp.float32)
        post_mask = query_valid_flat * (query_idx_raw_flat >= switch_idx).astype(jnp.float32)
        first_supervised_idx = jnp.min(
            jnp.where(
                query_valid_flat > 0.0,
                query_idx_raw_flat.astype(jnp.float32),
                total_steps_f,
            )
        )
        metrics = {
            "loss": total_loss,
            "teacher_loss": _masked_mean(teacher_per_sample, query_valid_flat),
            "zero_residual_loss": _masked_mean(endpoint_per_sample, query_valid_flat),
            "teacher_mae": _masked_mean(jnp.mean(jnp.abs(tau_err), axis=(-1, -2)), query_valid_flat),
            "teacher_rmse": jnp.sqrt(
                jnp.maximum(_masked_mean(jnp.mean(jnp.square(tau_err), axis=(-1, -2)), query_valid_flat), 0.0)
            ),
            "zero_residual_rms": jnp.sqrt(
                jnp.maximum(
                    _masked_mean(jnp.mean(jnp.square(endpoint_delta_err), axis=-1), query_valid_flat),
                    0.0,
                )
            ),
            "endpoint_residual_rms": jnp.sqrt(
                jnp.maximum(
                    _masked_mean(jnp.mean(jnp.square(endpoint_delta_pred), axis=-1), query_valid_flat),
                    0.0,
                )
            ),
            "delta_tau_rms": jnp.sqrt(
                jnp.maximum(
                    _masked_mean(jnp.mean(jnp.square(behavior_delta_flat), axis=-1), query_valid_flat),
                    0.0,
                )
            ),
            "tam_residual_rms": jnp.sqrt(
                jnp.maximum(
                    _masked_mean(jnp.mean(jnp.square(tam_residual_flat), axis=-1), query_valid_flat),
                    0.0,
                )
            ),
            "pre_switch_loss": _masked_mean(loss_per_sample, pre_mask),
            "post_switch_loss": _masked_mean(loss_per_sample, post_mask),
            "has_embedding_ratio": jnp.mean(trace["has_embedding"]),
            "supervised_ratio": supervised_count / total_steps_f,
            "q_tracking_rmse": jnp.sqrt(
                jnp.maximum(
                    jnp.sum(jnp.square(trace["q_cur"] - trace["q_ref"]))
                    / jnp.maximum(total_steps_f * dof_f, 1.0),
                    0.0,
                )
            ),
            "switch_idx": switch_idx.astype(jnp.float32),
            "switch_time_s": switch_idx.astype(jnp.float32) * float(cfg.dt),
            "first_supervised_idx": first_supervised_idx,
            "first_supervised_time_s": first_supervised_idx * float(cfg.dt),
            "num_supervised": supervised_count,
            "num_history_tokens": jnp.asarray(float(n_tokens), dtype=jnp.float32),
            "query_samples_per_token": jnp.asarray(float(query_samples_per_token), dtype=jnp.float32),
            "query_window_strides": jnp.asarray(float(query_window_strides), dtype=jnp.float32),
            "query_window_steps": jnp.asarray(float(query_window_steps), dtype=jnp.float32),
            "tau_map_sample_no": jnp.asarray(float(tau_map_sample_no), dtype=jnp.float32),
            "attention_history_tokens": jnp.asarray(
                float(history_train_window or 0),
                dtype=jnp.float32,
            ),
            "base_correction_alpha_mean": _masked_mean(jnp.mean(correction_alpha, axis=-1), query_valid_flat),
            "base_correction_alpha_max": _masked_mean(jnp.max(correction_alpha, axis=-1), query_valid_flat),
            "behavior_alpha_pre_switch": behavior_alpha_pair[0],
            "behavior_alpha_post_switch": behavior_alpha_pair[1],
            "behavior_alpha_mode_binary": jnp.asarray(
                1.0 if behavior_alpha_mode == "binary" else 0.0,
                dtype=jnp.float32,
            ),
            "behavior_alpha_one_prob": jnp.asarray(behavior_alpha_one_prob, dtype=jnp.float32),
            "behavior_torque_noise_scale": jnp.asarray(behavior_noise_scale, dtype=jnp.float32),
            "external_force_num_impulses": jnp.asarray(float(num_force_impulses), dtype=jnp.float32),
            "external_force_has_position_box": jnp.asarray(
                1.0 if force_has_position_box else 0.0,
                dtype=jnp.float32,
            ),
            "external_force_body_id": force_body_id.astype(jnp.float32),
            "supervision_alpha_mode_behavior": jnp.asarray(
                1.0 if supervision_alpha_mode == "behavior" else 0.0,
                dtype=jnp.float32,
            ),
            "tau_ref_noise_std": _tau_ref_noise_std(rollout_cmd_noise_std, tau_base_flat.dtype),
            "history_torque_mode_base_tam_fusion": jnp.asarray(
                1.0 if history_torque_mode == "base_tam_fusion" else 0.0,
                dtype=jnp.float32,
            ),
        }
        metrics = jax.tree.map(_finite_or_zero, metrics)
        return total_loss, metrics

    def batch_loss(params: Any, rng: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        keys = jax.random.split(rng, int(cfg.batch_size))
        losses, metrics = jax.vmap(single_episode_loss, in_axes=(None, 0))(params, keys)
        metrics_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)
        metrics_mean["loss_batch_std"] = jnp.std(losses)
        metrics_mean["switch_time_min_s"] = jnp.min(metrics["switch_time_s"])
        metrics_mean["switch_time_max_s"] = jnp.max(metrics["switch_time_s"])
        return jnp.mean(losses), metrics_mean

    return batch_loss


def build_exp4_periodic_eval(
    *,
    cfg: OnlineDaggerConfig,
    base_cfg: Any,
    hist_model: Any,
    adaptor_model: Any,
    mjx_model: mjx.Model,
    runtime: Any,
    profile_kwargs: dict[str, Any],
    external_force_body_id: int,
    arm_ids_np: np.ndarray,
) -> Optional[Any]:
    if int(cfg.exp4_eval_interval) <= 0:
        return None

    arm_ids = jnp.asarray(arm_ids_np, dtype=jnp.int32)
    dof = int(arm_ids_np.shape[0])
    dt = float(cfg.dt)
    hist_steps = int(round(float(cfg.exp4_eval_history_s) / dt))
    test_steps = int(round(float(cfg.exp4_eval_test_window_s) / dt))
    total_steps = hist_steps + test_steps
    if hist_steps <= 0 or test_steps <= 0 or total_steps < 2:
        raise ValueError(
            "Exp4 eval requires positive history/test windows; "
            f"got history={cfg.exp4_eval_history_s}, test={cfg.exp4_eval_test_window_s}."
        )
    num_tests = max(int(cfg.exp4_eval_num_tests), 1)
    adaptor_seq_length = _resolve_tam_seq_length(base_cfg)
    history_torque_mode = str(cfg.history_torque_mode)
    q_noise_std_rad = float(np.deg2rad(float(cfg.exp4_eval_q_noise_std_deg)))
    qd_noise_std_rad_s = float(np.deg2rad(float(cfg.exp4_eval_qd_noise_std_deg_s)))
    hidden_force_enabled = bool(int(cfg.exp4_eval_external_force_num_impulses) > 0 and external_force_body_id >= 0)
    norm_stats_ref = None

    sample_random_param_kwargs = {
        k: v for k, v in profile_kwargs.items() if k in _SAMPLE_RANDOM_PARAM_PROFILE_KEYS
    }
    waypoint_max_delta_deg_profile = profile_kwargs.get("waypoint_max_delta_deg_profile", None)
    external_force_magnitude_min_n = float(
        profile_kwargs.get("external_force_magnitude_min_n", cfg.exp4_eval_external_force_min_n)
    )
    external_force_magnitude_max_n = float(
        profile_kwargs.get("external_force_magnitude_max_n", cfg.exp4_eval_external_force_max_n)
    )
    actuator_trnid = jnp.asarray(mjx_model.actuator_trnid, dtype=jnp.int32)
    jnt_dofadr = jnp.asarray(mjx_model.jnt_dofadr, dtype=jnp.int32)
    act_jnt_id = actuator_trnid[:, 0]
    act_jnt_id_clamped = jnp.clip(act_jnt_id, 0, int(mjx_model.njnt) - 1)
    actuator_dof_abs = jnp.where(
        (act_jnt_id >= 0) & (act_jnt_id < int(mjx_model.njnt)),
        jnt_dofadr[act_jnt_id_clamped],
        jnp.minimum(jnp.arange(int(mjx_model.nu), dtype=jnp.int32), int(mjx_model.nv) - 1),
    )
    joint_range = jnp.asarray(mjx_model.jnt_range, dtype=jnp.float32)[arm_ids]
    data_template = mjx.make_data(mjx_model)
    mode_nominal = 0
    mode_tam = 1
    mode_teacher_ab = 2
    mode_teacher_ab_plus_tam = 3

    def step_dynamics(
        data: mjx.Data,
        tau_cmd: jax.Array,
        rollout_params: structs.RolloutParams,
        external_force_t: jax.Array,
    ) -> mjx.Data:
        qpos_for_act = data.qpos[actuator_dof_abs]
        qvel_for_act = data.qvel[actuator_dof_abs]
        tau_eff = actuator_util.actuator_model(
            tau_cmd,
            qpos_for_act,
            qvel_for_act,
            rollout_params.actuator_params,
        )
        ctrl_full = jnp.zeros((int(mjx_model.nu),), dtype=tau_eff.dtype).at[: tau_eff.shape[0]].set(tau_eff)
        model_step = rollout_params.set_mjx_model(mjx_model)
        xfrc_applied = jnp.zeros_like(data.xfrc_applied)
        if hidden_force_enabled:
            force_t = jnp.asarray(external_force_t, dtype=xfrc_applied.dtype)
            xfrc_applied = xfrc_applied.at[int(external_force_body_id), :3].set(force_t[:3])
        return mjx.step(model_step, data.replace(ctrl=ctrl_full, xfrc_applied=xfrc_applied))

    def sample_params(key: jax.Array) -> tuple[structs.RolloutParams, structs.RolloutParams]:
        _, pert_a_dict, _ = datagen.sample_random_params(
            key,
            mjx_model,
            evaluation_mode=True,
            **sample_random_param_kwargs,
        )
        _, pert_b_dict, _ = datagen.sample_random_params(
            jax.random.fold_in(key, 1),
            mjx_model,
            evaluation_mode=True,
            **sample_random_param_kwargs,
        )
        params_a = structs.RolloutParams(**pert_a_dict).fit_model_size(mjx_model)
        params_b = structs.RolloutParams(**pert_b_dict).fit_model_size(mjx_model)
        params_b = _replace_kp_kd(params_b, params_a.kp, params_a.kd)
        return params_a, params_b

    def make_eval_inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        k_wps, k_force, k_qn, k_dqn = jax.random.split(key, 4)
        waypoints = rollout.generate_waypoints(
            k_wps,
            int(cfg.exp4_eval_num_waypoints),
            batch_n=1,
            dof=dof,
            joint_range=joint_range,
            pause_prob=float(cfg.exp4_eval_pause_prob),
            waypoint_max_delta_deg_profile=waypoint_max_delta_deg_profile,
        )[0]
        q_ref, qd_ref = rollout.build_traj_from_waypoints(
            waypoints,
            total_steps,
            total_steps * dt,
        )
        if hidden_force_enabled:
            force_seq = rollout.sample_external_force_impulses(
                k_force,
                batch_n=1,
                total_steps=total_steps,
                dt=dt,
                num_impulses=int(cfg.exp4_eval_external_force_num_impulses),
                magnitude_min_n=external_force_magnitude_min_n,
                magnitude_max_n=external_force_magnitude_max_n,
                duration_min_s=float(cfg.exp4_eval_external_force_duration_min_s),
                duration_max_s=float(cfg.exp4_eval_external_force_duration_max_s),
                dtype=jnp.float32,
            )[0]
        else:
            force_seq = jnp.zeros((total_steps, 3), dtype=jnp.float32)
        q_noise = q_noise_std_rad * jax.random.normal(k_qn, (total_steps, dof), dtype=jnp.float32)
        qd_noise = qd_noise_std_rad_s * jax.random.normal(k_dqn, (total_steps, dof), dtype=jnp.float32)
        return q_ref, qd_ref, force_seq, q_noise, qd_noise

    def single_eval(model_params: Any, norm_stats: Any, key: jax.Array) -> dict[str, jax.Array]:
        key_params, key_inputs = jax.random.split(key)
        params_a, params_b = sample_params(key_params)
        q_ref, qd_ref, force_seq, q_noise_seq, qd_noise_seq = make_eval_inputs(key_inputs)
        norm_stats_aligned = align_norm_stats_to_dof(norm_stats, dof)
        controller_fn = params_a.controller_params.get_actuator_fn(
            control_type="qref",
            ideal_mjx_model=mjx_model,
            add_noise=False,
        )

        def run_curve(curve_mode: int) -> tuple[jax.Array, jax.Array]:
            data0 = data_template.replace(
                qpos=jnp.zeros((int(mjx_model.nq),), dtype=jnp.float32).at[arm_ids].set(q_ref[0]),
                qvel=jnp.zeros((int(mjx_model.nv),), dtype=jnp.float32).at[arm_ids].set(qd_ref[0]),
                ctrl=jnp.zeros((int(mjx_model.nu),), dtype=jnp.float32),
                xfrc_applied=jnp.zeros_like(data_template.xfrc_applied),
            )
            q_win0 = jnp.repeat(q_ref[0][None, :], adaptor_seq_length, axis=0)
            qd_win0 = jnp.repeat(qd_ref[0][None, :], adaptor_seq_length, axis=0)
            tau_win0 = jnp.zeros((adaptor_seq_length, dof), dtype=jnp.float32)
            ctrl_carry0 = jnp.zeros((1, dof), dtype=jnp.float32)
            online_state_applied0 = init_online_history_state(runtime, dtype=jnp.float32)
            online_state_base0 = init_online_history_state(runtime, dtype=jnp.float32)
            online_state_tam0 = init_online_history_state(runtime, dtype=jnp.float32)

            def scan_step(carry, inp):
                data, q_win, qd_win, tau_win, ctrl_carry, state_applied, state_base, state_tam = carry
                abs_idx, q_ref_t, qd_ref_t, force_t, q_noise_t, qd_noise_t = inp
                use_b = abs_idx >= hist_steps
                active_params = _select_rollout_params(use_b, params_a, params_b)
                q_cur = data.qpos[arm_ids]
                qd_cur = data.qvel[arm_ids]
                q_obs = q_cur + q_noise_t
                qd_obs = qd_cur + qd_noise_t
                q_win = push_window(q_win, q_obs)
                qd_win = push_window(qd_win, qd_obs)
                tau_plain_b, _, ctrl_carry = controller_fn(
                    q_win[None, ...],
                    qd_win[None, ...],
                    q_ref_t[None, :],
                    qd_ref_t[None, :],
                    jax.random.PRNGKey(0),
                    ctrl_carry,
                    u_ref=None,
                )
                tau_plain = tau_plain_b[0]
                teacher_base = compute_gt_tau_cmd(
                    mjx_model,
                    active_params,
                    q_obs,
                    qd_obs,
                    tau_plain,
                    method="shared_linear",
                )

                def tam_with_base(base_tau):
                    tau_win_for = push_window(tau_win, base_tau)
                    history_emb = _history_embedding_for_mode(
                        model_params,
                        history_torque_mode,
                        state_applied.history_emb,
                        state_base.history_emb,
                        state_tam.history_emb,
                    )
                    tau_candidate, _ = apply_online_adaptor(
                        adaptor_model=adaptor_model,
                        adaptor_apply_fn=None,
                        params_adaptor=model_params["adaptor"],
                        q_window=q_win,
                        qd_window=qd_win,
                        tau_window=tau_win_for,
                        history_emb=history_emb,
                        norm_stats=norm_stats_aligned,
                    )
                    has_z = state_applied.has_embedding
                    tau_out = jnp.where(has_z, tau_candidate, base_tau)
                    return tau_out, tau_out - base_tau

                zero_delta = jnp.zeros_like(tau_plain)
                if curve_mode == mode_nominal:
                    tau_cmd, delta_tau = tau_plain, zero_delta
                elif curve_mode == mode_teacher_ab:
                    tau_cmd, delta_tau = teacher_base, zero_delta
                elif curve_mode == mode_teacher_ab_plus_tam:
                    tau_cmd, delta_tau = tam_with_base(teacher_base)
                else:
                    tau_cmd, delta_tau = tam_with_base(tau_plain)

                data_next = step_dynamics(data, tau_cmd, active_params, force_t)
                behavior_delta = tau_cmd - tau_plain
                tam_residual = delta_tau
                tau_win_next = push_window(tau_win, tau_cmd)
                state_applied_next = advance_online_history_state(
                    runtime,
                    state_applied,
                    q_arm=q_obs,
                    qd_arm=qd_obs,
                    tau_arm=tau_cmd,
                    raw_tau_arm=tau_cmd,
                    params_hist=model_params["hist"],
                    norm_stats=norm_stats_aligned,
                )
                state_base_next = advance_online_history_state(
                    runtime,
                    state_base,
                    q_arm=q_obs,
                    qd_arm=qd_obs,
                    tau_arm=tau_plain,
                    raw_tau_arm=tau_cmd,
                    params_hist=model_params["hist"],
                    norm_stats=norm_stats_aligned,
                )
                state_tam_next = advance_online_history_state(
                    runtime,
                    state_tam,
                    q_arm=q_obs,
                    qd_arm=qd_obs,
                    tau_arm=tam_residual,
                    raw_tau_arm=tau_cmd,
                    params_hist=model_params["hist"],
                    norm_stats=norm_stats_aligned,
                )
                return (
                    data_next,
                    q_win,
                    qd_win,
                    tau_win_next,
                    ctrl_carry,
                    state_applied_next,
                    state_base_next,
                    state_tam_next,
                ), (q_cur, delta_tau)

            (data1, *_), (q_log, delta_log) = jax.lax.scan(
                scan_step,
                (
                    data0,
                    q_win0,
                    qd_win0,
                    tau_win0,
                    ctrl_carry0,
                    online_state_applied0,
                    online_state_base0,
                    online_state_tam0,
                ),
                (
                    jnp.arange(total_steps, dtype=jnp.int32),
                    q_ref,
                    qd_ref,
                    force_seq,
                    q_noise_seq,
                    qd_noise_seq,
                ),
            )
            del data1
            return q_log, delta_log

        q_nominal, delta_nominal = run_curve(mode_nominal)
        q_tam, delta_tam = run_curve(mode_tam)
        q_teacher_ab, delta_teacher_ab = run_curve(mode_teacher_ab)
        q_teacher_plus_tam, delta_teacher_plus_tam = run_curve(mode_teacher_ab_plus_tam)

        def span_rmse_deg(q_trace: jax.Array, start: int, end: int) -> jax.Array:
            start_i = int(max(0, min(start, total_steps - 1)))
            end_i = int(max(start_i + 1, min(end, total_steps)))
            err = q_trace[start_i:end_i] - q_ref[start_i:end_i]
            return jnp.sqrt(jnp.mean(jnp.square(err))) * (180.0 / jnp.pi)

        def span_delta_rms(delta_trace: jax.Array, start: int, end: int) -> jax.Array:
            start_i = int(max(0, min(start, total_steps - 1)))
            end_i = int(max(start_i + 1, min(end, total_steps)))
            return jnp.sqrt(jnp.mean(jnp.square(delta_trace[start_i:end_i])))

        post0_end = min(total_steps, hist_steps + int(round(1.0 / dt)))
        post2_start = min(total_steps - 1, hist_steps + int(round(2.0 / dt)))
        post6_end = min(total_steps, hist_steps + int(round(6.0 / dt)))
        base_tam_steady_rmse = span_rmse_deg(q_tam, post2_start, post6_end)
        teacher_tam_steady_rmse = span_rmse_deg(q_teacher_plus_tam, post2_start, post6_end)
        metrics = {
            "exp4/nominal_q_rmse_deg": span_rmse_deg(q_nominal, hist_steps, total_steps),
            "exp4/tam_q_rmse_deg": span_rmse_deg(q_tam, hist_steps, total_steps),
            "exp4/teacher_ab_q_rmse_deg": span_rmse_deg(q_teacher_ab, hist_steps, total_steps),
            "exp4/teacher_plus_tam_q_rmse_deg": span_rmse_deg(q_teacher_plus_tam, hist_steps, total_steps),
            "exp4/tam_q_rmse_0_1s_deg": span_rmse_deg(q_tam, hist_steps, post0_end),
            "exp4/teacher_ab_q_rmse_0_1s_deg": span_rmse_deg(q_teacher_ab, hist_steps, post0_end),
            "exp4/teacher_plus_tam_q_rmse_0_1s_deg": span_rmse_deg(q_teacher_plus_tam, hist_steps, post0_end),
            "exp4/tam_q_rmse_2_6s_deg": base_tam_steady_rmse,
            "exp4/teacher_ab_q_rmse_2_6s_deg": span_rmse_deg(q_teacher_ab, post2_start, post6_end),
            "exp4/teacher_plus_tam_q_rmse_2_6s_deg": teacher_tam_steady_rmse,
            "exp4/base_tam_steady_q_rmse_deg": base_tam_steady_rmse,
            "exp4/teacher_tam_steady_q_rmse_deg": teacher_tam_steady_rmse,
            "exp4/tam_delta_rms": span_delta_rms(delta_tam, hist_steps, total_steps),
            "exp4/teacher_plus_tam_delta_rms": span_delta_rms(delta_teacher_plus_tam, hist_steps, total_steps),
            "exp4/teacher_ab_delta_rms": span_delta_rms(delta_teacher_ab, hist_steps, total_steps),
            "exp4/nominal_delta_rms": span_delta_rms(delta_nominal, hist_steps, total_steps),
        }
        return jax.tree.map(_finite_or_zero, metrics)

    def eval_batch(model_params: Any, norm_stats: Any, rng: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(rng, num_tests)
        metrics = jax.vmap(single_eval, in_axes=(None, None, 0))(model_params, norm_stats, keys)
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)

    del norm_stats_ref
    return jax.jit(eval_batch)


def main(cfg: OnlineDaggerConfig) -> None:
    _require_checkpoint(cfg)

    inference = SimAdaptorInference(
        cfg.ckpt,
        checkpoint_step=cfg.checkpoint_step,
        xml_path=cfg.xml,
    )
    inference._ensure_checkpoint_loaded()
    base_cfg = inference._cfg
    hist_model, adaptor_model = inference._simadaptor_model
    params0 = inference._simadaptor_params
    norm_stats = inference._norm_stats
    mjx_model = inference._mjx_model_template
    xml_path = Path(inference.xml_path).expanduser().resolve()

    arm_ids_np = np.asarray(rollout.guess_arm_joint_ids(mujoco.MjModel.from_xml_path(str(xml_path)), dof_target=7))
    dof = int(arm_ids_np.shape[0])
    runtime = build_online_history_runtime(
        hist_model=hist_model,
        params_hist_example=params0["hist"],
        emb_dim=int(getattr(base_cfg, "emb_dim")),
        arm_dof=dof,
        attention_history_s=cfg.attention_history_s,
        sample_dt_s=cfg.dt,
    )
    if str(cfg.history_torque_mode) == "base_tam_fusion":
        params0 = _ensure_history_fusion_params(params0, emb_dim=int(getattr(base_cfg, "emb_dim")))

    if cfg.loss_stride_steps is None:
        cfg.loss_stride_steps = int(runtime.config.patch_stride)
    robot_key = cfg.profile_key or derive_robot_key(xml_path)
    resolved_profile_key, profile_kwargs = load_datagen_profile(cfg.profile_table, robot_key)
    (
        external_force_body_ids_np,
        external_force_body_names,
        external_force_position_min_np,
        external_force_position_max_np,
    ) = _resolve_external_force_targets(
        xml_path,
        body_name=cfg.external_force_body_name,
        profile_kwargs=profile_kwargs,
        use_profile_targets=bool(cfg.external_force_use_profile_targets),
    )
    external_force_body_id = int(external_force_body_ids_np[0]) if len(external_force_body_ids_np) else -1
    external_force_body_name = external_force_body_names[0] if external_force_body_names else "none"
    external_force_targets_label = ",".join(
        f"{name}({int(body_id)})"
        for name, body_id in zip(external_force_body_names, external_force_body_ids_np)
    )
    if not external_force_targets_label:
        external_force_targets_label = f"{external_force_body_name}({external_force_body_id})"

    run_name = cfg.run_name or f"tam_online_dagger_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg.workdir).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        out_dir,
        base_cfg=base_cfg,
        dagger_cfg=cfg,
        params=params0,
        norm_stats=norm_stats,
        source_checkpoint=inference._ckpt_path,
        xml_path=xml_path,
    )

    behavior_rollout_label = (
        "base_plus_tam_residual"
        if float(cfg.behavior_alpha_min) == 0.0 and float(cfg.behavior_alpha_max) == 0.0
        else "teacher_alpha_plus_tam_residual"
    )
    print(
        "[online-dagger] "
        f"out={out_dir} batch={cfg.batch_size} steps={cfg.max_steps} "
        f"profile={resolved_profile_key} "
        f"episode={cfg.episode_duration_s}s patch={runtime.config.patch_size} "
        f"stride={runtime.config.patch_stride} context_half={runtime.config.context_half} "
        f"attention_history_s={cfg.attention_history_s} "
        f"attention_history_tokens={runtime.config.attention_history_tokens} "
        f"loss_stride={cfg.loss_stride_steps} query_samples_per_token={cfg.query_samples_per_token} "
        f"query_window_strides={cfg.query_window_strides} "
        f"tau_map_sample_no={cfg.tau_map_sample_no} history_torque_mode={cfg.history_torque_mode} "
        f"behavior_alpha_mode={cfg.behavior_alpha_mode} "
        f"behavior_alpha_one_prob={cfg.behavior_alpha_one_prob} "
        f"behavior_rollout={behavior_rollout_label} "
        f"supervision_alpha_mode={cfg.supervision_alpha_mode} "
        f"behavior_torque_noise_scale={cfg.behavior_torque_noise_scale} "
        f"exp4_eval_interval={cfg.exp4_eval_interval} "
        f"exp4_eval_num_tests={cfg.exp4_eval_num_tests} "
        f"exp4_eval_windows={cfg.exp4_eval_history_s}+{cfg.exp4_eval_test_window_s}s "
        f"external_force_targets={external_force_targets_label} "
        f"external_force_profile_targets={bool(cfg.external_force_use_profile_targets)} "
        f"external_force_position_boxes={external_force_position_min_np is not None}"
    )

    tx = _build_optimizer(cfg)
    state = OnlineDaggerState.create(
        apply_fn=None,
        params=params0,
        tx=tx,
        norm_stats=norm_stats,
    )
    loss_fn = build_loss(
        cfg=cfg,
        base_cfg=base_cfg,
        hist_model=hist_model,
        adaptor_model=adaptor_model,
        mjx_model=mjx_model,
        runtime=runtime,
        norm_stats=norm_stats,
        profile_kwargs=profile_kwargs,
        external_force_body_ids_np=external_force_body_ids_np,
        external_force_position_min_np=external_force_position_min_np,
        external_force_position_max_np=external_force_position_max_np,
        arm_ids_np=arm_ids_np,
    )
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    exp4_eval_fn = build_exp4_periodic_eval(
        cfg=cfg,
        base_cfg=base_cfg,
        hist_model=hist_model,
        adaptor_model=adaptor_model,
        mjx_model=mjx_model,
        runtime=runtime,
        profile_kwargs=profile_kwargs,
        external_force_body_id=external_force_body_id,
        arm_ids_np=arm_ids_np,
    )

    rng = jax.random.PRNGKey(int(cfg.seed))
    if cfg.debug_grad_check:
        rng, step_key = jax.random.split(rng)
        (_, debug_metrics), grads = loss_and_grad(state.params, step_key)
        hist_norm = optax.global_norm(grads.get("hist", {}))
        adaptor_norm = optax.global_norm(grads.get("adaptor", {}))
        fusion_norm = optax.global_norm(grads.get("history_fusion", {}))
        print(
            "[online-dagger] grad-check "
            f"hist_norm={_to_float(hist_norm):.6g} adaptor_norm={_to_float(adaptor_norm):.6g} "
            f"fusion_norm={_to_float(fusion_norm):.6g} "
            f"loss={_to_float(debug_metrics['loss']):.6g}"
        )
        if not np.isfinite(_to_float(hist_norm)) or not np.isfinite(_to_float(adaptor_norm)):
            raise RuntimeError("Gradient check produced non-finite hist/adaptor gradients.")
        if _to_float(hist_norm) <= 0.0 or _to_float(adaptor_norm) <= 0.0:
            raise RuntimeError("Gradient check expected non-zero hist and adaptor gradients.")
        if str(cfg.history_torque_mode) == "base_tam_fusion":
            if not np.isfinite(_to_float(fusion_norm)) or _to_float(fusion_norm) <= 0.0:
                raise RuntimeError("Gradient check expected non-zero history_fusion gradients.")

    use_wandb = str(cfg.wandb_mode).lower() != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            config=json.loads(json.dumps(dataclasses.asdict(cfg), default=_json_default)),
        )

    try:
        for step in range(1, int(cfg.max_steps) + 1):
            rng, step_key = jax.random.split(rng)
            (loss_value, metrics), grads = loss_and_grad(state.params, step_key)
            state = state.apply_gradients(grads=grads)

            if step == 1 or step % int(cfg.log_interval) == 0:
                grad_norm = optax.global_norm(grads)
                log_metrics = {k: _to_float(v) for k, v in metrics.items()}
                log_metrics["loss"] = _to_float(loss_value)
                log_metrics["grad_norm"] = _to_float(grad_norm)
                print(
                    f"[online-dagger] step={step} "
                    f"loss={log_metrics['loss']:.6f} "
                    f"teacher_rmse={log_metrics['teacher_rmse']:.4f} "
                    f"zero_rms={log_metrics['zero_residual_rms']:.4f} "
                    f"q_rmse={log_metrics['q_tracking_rmse']:.5f} "
                    f"has_z={log_metrics['has_embedding_ratio']:.3f} "
                    f"switch={log_metrics['switch_time_s']:.3f}s"
                )
                if use_wandb:
                    wandb.log(log_metrics, step=step)

            should_run_exp4_eval = exp4_eval_fn is not None and (
                (bool(cfg.exp4_eval_warmup) and step == 1)
                or (int(cfg.exp4_eval_interval) > 0 and step % int(cfg.exp4_eval_interval) == 0)
            )
            if should_run_exp4_eval:
                rng, eval_key = jax.random.split(rng)
                eval_metrics_raw = jax.block_until_ready(
                    exp4_eval_fn(state.params, state.norm_stats, eval_key)
                )
                eval_metrics = {k: _to_float(v) for k, v in eval_metrics_raw.items()}
                print(
                    f"[online-dagger][exp4-eval] step={step} "
                    f"tam_rmse={eval_metrics['exp4/tam_q_rmse_deg']:.4f}deg "
                    f"teacher_ab_rmse={eval_metrics['exp4/teacher_ab_q_rmse_deg']:.4f}deg "
                    f"teacher_plus_tam_rmse={eval_metrics['exp4/teacher_plus_tam_q_rmse_deg']:.4f}deg "
                    f"base_tam_steady={eval_metrics['exp4/base_tam_steady_q_rmse_deg']:.4f}deg "
                    f"teacher_tam_steady={eval_metrics['exp4/teacher_tam_steady_q_rmse_deg']:.4f}deg "
                    f"teacher_plus_tam_delta={eval_metrics['exp4/teacher_plus_tam_delta_rms']:.4f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log(eval_metrics, step=step)

            if cfg.ckpt_interval > 0 and step % int(cfg.ckpt_interval) == 0:
                _save_checkpoint(
                    out_dir,
                    state,
                    step=step,
                    cfg=cfg,
                    base_cfg=base_cfg,
                    source_checkpoint=inference._ckpt_path,
                    xml_path=xml_path,
                )
                print(f"[online-dagger] saved checkpoint step={step} -> {out_dir}")
    finally:
        if cfg.save_final:
            _save_checkpoint(
                out_dir,
                state,
                step=int(state.step),
                cfg=cfg,
                base_cfg=base_cfg,
                source_checkpoint=inference._ckpt_path,
                xml_path=xml_path,
            )
            print(f"[online-dagger] saved final checkpoint step={int(state.step)} -> {out_dir}")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main(parse_tyro_config(OnlineDaggerConfig))
