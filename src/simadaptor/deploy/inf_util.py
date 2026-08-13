import os
from collections.abc import Mapping
import dataclasses
import pickle
import json
import warnings
from pathlib import Path

import jax
import tqdm
from simadaptor.physics import smoothing as smoothing_util
import simadaptor.models.adaptor as models
import simadaptor.models.transformer as models_transformer
import simadaptor.physics.rollout as rollout
import simadaptor.core.structs as structs
import jax.numpy as jnp
from mujoco import mjx
import mujoco
import copy
import numpy as np
import simadaptor.physics.dynamics as dynamics
import simadaptor.physics.actuator as actuator_util
from simadaptor.config import TrainConfig
from simadaptor.deploy.runtime_common import (
    HistoryRuntimeBundle,
    prepare_history_inputs,
)


DEFAULT_ADAPTOR_DEADZONE_THRESHOLDS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32
)

_ADAPTOR_FLAG_JOINTWISE_CONDITIONING = 1 << 1
_ADAPTOR_FLAG_JOINTWISE_DIRECT_RESIDUAL_HEAD = 1 << 3
_ADAPTOR_FLAG_COMMAND_CONDITIONED_HEAD = 1 << 4


class _OnlineDaggerConfigCompat:
    """Stable target for legacy ``__main__.OnlineDaggerConfig`` pickles.

    Online DAgger checkpoints were written while the trainer was executed as a
    script, so pickle recorded the config class under ``__main__``.  Deployment
    entrypoints have a different ``__main__`` module and otherwise cannot load
    those checkpoints.  The config is metadata-only at inference time, so a
    state-compatible attribute container is sufficient and keeps the original
    values available for checkpoint-driven deployment decisions.
    """


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "OnlineDaggerConfig" and (
            module in ("__main__", "__main___impl")
            or module == "train_tam_online_dagger"
            or module.endswith(".train_tam_online_dagger")
            or module == "dagger_finetune_impl"
            or module.endswith("dagger_finetune_impl")
        ):
            return _OnlineDaggerConfigCompat
        return super().find_class(module, name)


def _load_viser_util():
    raise RuntimeError(
        "Viser visualization support is not included in the minimal TAM release."
    )


def _extract_restored_field(restored, field: str, default=None):
    """Read a field from an object or mapping returned by checkpoint restore."""
    if hasattr(restored, field):
        return getattr(restored, field)
    if isinstance(restored, Mapping):
        return restored.get(field, default)
    return default


def _checkpoint_step_from_dir_name(path: Path) -> int | None:
    """Parse `checkpoint_<step>` directory names used by Flax/Orbax."""
    name = path.name
    if not name.startswith("checkpoint_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


_LEGACY_TAM_CFG_FIELD_MAP = (
    ("adaptor_hidden", "tam_hidden"),
    ("adaptor_depth", "tam_depth"),
    ("adaptor_seq_length", "tam_seq_length"),
)

_COMPATIBLE_CHECKPOINT_ABLATION_MODES = ("tam", "full_mam")


def _migrate_legacy_cfg_fields(cfg_obj: object) -> None:
    """Promote raw pickled legacy field values on restored checkpoint configs.

    Configs pickled before the tam_* field rename carry their trained values in
    the instance __dict__ under the legacy names, where the class-level compat
    properties shadow them on attribute access. Copy them into the live tam_*
    fields so model construction sees the trained values instead of class
    defaults, and reject checkpoints whose raw training mode is incompatible
    with public TAM inference.
    """
    inst = getattr(cfg_obj, "__dict__", None)
    if not isinstance(inst, dict):
        return
    for legacy_name, field_name in _LEGACY_TAM_CFG_FIELD_MAP:
        if legacy_name in inst and field_name not in inst:
            try:
                setattr(cfg_obj, field_name, int(inst[legacy_name]))
            except (TypeError, ValueError):
                pass
    raw_mode = inst.get("ablation_mode")
    if raw_mode is not None and str(raw_mode) not in _COMPATIBLE_CHECKPOINT_ABLATION_MODES:
        raise ValueError(
            "Public TAM inference supports only TAM checkpoints; this checkpoint "
            f"was trained with ablation_mode={str(raw_mode)!r}."
        )


def _backfill_missing_cfg_fields(cfg_obj: object, defaults: object) -> None:
    """Backfill newly-added dataclass fields on old pickled config objects."""
    legacy_field_defaults = {
        "ablation_mode": "tam",
        "random_input_delay_enable": True,
    }
    if cfg_obj is None or defaults is None:
        return
    for field in dataclasses.fields(defaults):
        name = field.name
        if not hasattr(cfg_obj, name):
            if name in legacy_field_defaults:
                setattr(cfg_obj, name, copy.deepcopy(legacy_field_defaults[name]))
            else:
                setattr(cfg_obj, name, copy.deepcopy(getattr(defaults, name)))
            continue
        cur_val = getattr(cfg_obj, name)
        default_val = getattr(defaults, name)
        if dataclasses.is_dataclass(default_val) and cur_val is not None:
            _backfill_missing_cfg_fields(cur_val, default_val)


def adaptor_deadzone_scaling(u_des, delta_tau, deadzone_thresholds=None):
    """Scale adaptor correction when desired torque lies inside a deadzone.

    Args:
        u_des: [..., DoF] desired torques for the final timestep.
        delta_tau: [..., DoF] adaptor torque correction to be scaled.
        deadzone_thresholds: optional array-like [DoF] specifying per-joint
            deadzone magnitudes. Defaults to DEFAULT_ADAPTOR_DEADZONE_THRESHOLDS.
    Returns:
        Scaled delta_tau tensor with reduced magnitude inside each joint's deadzone.
    """

    if deadzone_thresholds is None:
        deadzone_thresholds = DEFAULT_ADAPTOR_DEADZONE_THRESHOLDS

    u_des = jnp.asarray(u_des)
    delta_tau = jnp.asarray(delta_tau)
    deadzone_thresholds = jnp.asarray(deadzone_thresholds, dtype=u_des.dtype).reshape((-1,))
    target_dof = int(u_des.shape[-1])
    src_dof = int(deadzone_thresholds.shape[0])
    if src_dof == 0:
        thresholds = jnp.zeros((target_dof,), dtype=u_des.dtype)
    elif src_dof >= target_dof:
        thresholds = deadzone_thresholds[:target_dof]
    else:
        tail = jnp.repeat(deadzone_thresholds[-1:], target_dof - src_dof, axis=0)
        thresholds = jnp.concatenate([deadzone_thresholds, tail], axis=0)

    thresholds = jnp.clip(thresholds, 0.0, None)
    abs_u = jnp.abs(u_des)

    # Avoid division for zero thresholds (no scaling => factor 1.0).
    positive_mask = thresholds > 0.0
    safe_thresholds = jnp.where(positive_mask, thresholds, 1.0)
    ratio = abs_u / safe_thresholds
    ratio = jnp.clip(ratio, 0.0, 1.0)
    scaling = jnp.where(abs_u < safe_thresholds, ratio, 1.0)
    scaling = jnp.where(positive_mask, scaling, 1.0)

    return delta_tau * scaling

class SimAdaptorInference:
    @staticmethod
    def _resolve_checkpoint(
        simadaptor_ckpt_path=None,
        *,
        include_robot_assets: bool = True,
        checkpoint_step: int | None = None,
    ) -> Path:
        """
        Resolve a local checkpoint path. Explicit local paths may point to either
        a run directory containing save_dict.pkl or a specific Flax/Orbax
        `checkpoint_<step>` directory within that run.
        """
        del include_robot_assets
        if simadaptor_ckpt_path is None:
            raise ValueError("Provide simadaptor_ckpt_path.")
        if checkpoint_step is not None:
            raise ValueError("checkpoint_step is not supported for local checkpoint paths.")
        ckpt_dir = Path(simadaptor_ckpt_path).expanduser()
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        return ckpt_dir

    @staticmethod
    def _find_checkpoint_metadata_dir(ckpt_path: Path) -> Path | None:
        """Find the nearest ancestor directory that contains save_dict.pkl metadata."""
        for candidate in (ckpt_path, *ckpt_path.parents):
            if (candidate / "save_dict.pkl").exists():
                return candidate
        return None

    @staticmethod
    def _looks_like_flax_checkpoint_dir(ckpt_path: Path) -> bool:
        """Heuristic for Flax/Orbax step checkpoint directories."""
        if not ckpt_path.is_dir():
            return False
        if ckpt_path.name == "checkpoint" or ckpt_path.name.startswith("checkpoint_"):
            return True
        marker_names = (
            "_CHECKPOINT_METADATA",
            "_METADATA",
            "_sharding",
            "manifest.ocdbt",
        )
        return any((ckpt_path / marker).exists() for marker in marker_names)

    @staticmethod
    def _load_save_dict_payload(ckpt_dir: Path) -> dict:
        save_dict_path = ckpt_dir / "save_dict.pkl"
        with open(save_dict_path, "rb") as f:
            return _CheckpointMetadataUnpickler(f).load()

    @staticmethod
    def _restore_flax_checkpoint(ckpt_path: Path):
        try:
            from flax.training import checkpoints
        except ImportError as exc:
            raise RuntimeError(
                "Loading SimAdaptor from checkpoint_<step> requires flax to be installed."
            ) from exc

        restore_attempts: list[tuple[str, dict]] = [(str(ckpt_path), {})]
        step = _checkpoint_step_from_dir_name(ckpt_path)
        if step is not None:
            restore_attempts.append((str(ckpt_path.parent), {"step": step}))

        last_error = None
        for restore_dir, kwargs in restore_attempts:
            try:
                return checkpoints.restore_checkpoint(restore_dir, target=None, **kwargs)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to restore Flax checkpoint from {ckpt_path}: {last_error}") from last_error

    @staticmethod
    def _resolve_ckpt_xml_path(ckpt_dir: Path, cfg) -> Path | None:
        """Resolve robot XML from checkpoint contents/config without robot-specific defaults."""
        robot_dir = ckpt_dir / "robot_model"
        manifest_path = robot_dir / "manifest.json"
        candidates: list[Path] = []

        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                xml_rel = manifest.get("robot_xml")
                if xml_rel:
                    candidates.append((robot_dir / xml_rel).expanduser())
            except Exception as e:
                print(f"Warning: failed to read manifest {manifest_path}: {e}")

        # Common packaging fallback for exported robot bundle.
        candidates.append((robot_dir / "robot.xml").expanduser())

        cfg_xml_raw = None
        data_cfg = getattr(cfg, "data", None) if cfg is not None else None
        if data_cfg is not None and hasattr(data_cfg, "xml_path"):
            cfg_xml_raw = getattr(data_cfg, "xml_path")
        if cfg_xml_raw:
            cfg_xml = Path(cfg_xml_raw).expanduser()
            candidates.append(cfg_xml)
            if not cfg_xml.is_absolute():
                candidates.append((ckpt_dir / cfg_xml).expanduser())
                candidates.append((robot_dir / cfg_xml.name).expanduser())

        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        return None

    @staticmethod
    def _resolve_xml_path(
        user_xml_path: str | Path | None,
        ckpt_dir: Path | None,
        cfg,
    ) -> Path:
        """Resolve XML path from explicit CLI arg or checkpoint metadata."""
        if user_xml_path is not None:
            xml_path = Path(user_xml_path).expanduser()
            if not xml_path.exists():
                raise FileNotFoundError(f"XML not found: {xml_path}")
            return xml_path.resolve()

        if ckpt_dir is None:
            raise ValueError(
                "XML path is required when no checkpoint is provided. "
                "Pass xml_path explicitly."
            )

        ckpt_xml = SimAdaptorInference._resolve_ckpt_xml_path(ckpt_dir, cfg)
        if ckpt_xml is None:
            raise FileNotFoundError(
                "Could not resolve robot XML from checkpoint. "
                "Provide xml_path explicitly. This is required for multi-robot checkpoints without bundled robot assets."
            )
        return ckpt_xml

    def __init__(
        self,
        simadaptor_ckpt_path: str | Path | None = None,
        *,
        checkpoint_step: int | None = None,
        xml_path: str | Path | None = None,
        load_collision_models: bool = False,
    ):
        if simadaptor_ckpt_path is None:
            ckpt_dir = None
        else:
            ckpt_dir = self._resolve_checkpoint(
                simadaptor_ckpt_path,
                include_robot_assets=xml_path is None,
                checkpoint_step=checkpoint_step,
            )
        self._ckpt_path = Path(ckpt_dir).expanduser().resolve() if ckpt_dir is not None else None
        self._ckpt_metadata_path = (
            self._find_checkpoint_metadata_dir(self._ckpt_path)
            if self._ckpt_path is not None
            else None
        )

        self._cfg = None
        self._checkpoint_metadata_payload = {}
        self._simadaptor_model = None
        self._simadaptor_params = None
        self._norm_stats = None
        self._ideal_model_has_gravity = True
        self._load_collision_models_requested = bool(load_collision_models)
        self._collision_models_loaded = False
        self._collision_model_path = None
        self._mj_model_selfcol = None
        self._mjx_model_selfcol = None
        self._arm_joint_ids = None
        self._arm_qpos_idx = None
        self._arm_joint_range = None
        self._clearance_models = {}

        adaptor = None
        if self._ckpt_path is not None:
            if (self._ckpt_path / "save_dict.pkl").exists():
                loaded = self._load_save_dict_payload(self._ckpt_path)
                self._checkpoint_metadata_payload = dict(loaded)
                simadaptor_params = loaded["params"]
                self._norm_stats = loaded.get("norm_stats")
                cfg = loaded["cfg"]
                print(f"[SimAdaptorInference] Loaded checkpoint from {self._ckpt_path}")
            elif self._looks_like_flax_checkpoint_dir(self._ckpt_path):
                if self._ckpt_metadata_path is None:
                    raise FileNotFoundError(
                        "Could not find save_dict.pkl for intermediate checkpoint "
                        f"{self._ckpt_path}. Pass a run directory containing save_dict.pkl, "
                        "or keep the checkpoint inside its original run folder."
                    )
                metadata_loaded = self._load_save_dict_payload(self._ckpt_metadata_path)
                self._checkpoint_metadata_payload = dict(metadata_loaded)
                restored = self._restore_flax_checkpoint(self._ckpt_path)
                simadaptor_params = _extract_restored_field(restored, "params")
                if simadaptor_params is None:
                    raise ValueError(
                        "Restored Flax checkpoint does not contain `params`; "
                        f"got type {type(restored)!r}."
                    )
                self._norm_stats = _extract_restored_field(
                    restored, "norm_stats", metadata_loaded.get("norm_stats")
                )
                cfg = metadata_loaded.get("cfg")
                restored_step = _extract_restored_field(restored, "step")
                print(f"[SimAdaptorInference] Loaded intermediate checkpoint weights from {self._ckpt_path}")
                if restored_step is not None:
                    try:
                        print(f"[SimAdaptorInference] Restored training step: {int(np.asarray(restored_step))}")
                    except Exception:
                        print(f"[SimAdaptorInference] Restored training step: {restored_step}")
                print(
                    "[SimAdaptorInference] Loaded checkpoint config from "
                    f"{self._ckpt_metadata_path / 'save_dict.pkl'}"
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint path does not contain save_dict.pkl and does not look like "
                    f"a Flax checkpoint directory: {self._ckpt_path}"
                )

            if dataclasses.is_dataclass(cfg):
                try:
                    _backfill_missing_cfg_fields(cfg, TrainConfig())
                except Exception as exc:
                    print(f"[SimAdaptorInference] Warning: failed to backfill checkpoint config fields ({exc})")
            _migrate_legacy_cfg_fields(cfg)
            self._cfg = cfg
            print(f"[SimAdaptorInference] Config: {cfg}")
            if self._norm_stats is None:
                print("[SimAdaptorInference] Norm stats: not found")
            elif isinstance(self._norm_stats, dict):
                stat_keys = list(self._norm_stats.keys())
                shapes = {k: getattr(v, 'shape', None) for k, v in self._norm_stats.items()}
                print(f"[SimAdaptorInference] Norm stats loaded (dict): keys={stat_keys}, shapes={shapes}")
            elif dataclasses.is_dataclass(self._norm_stats):
                fields = dataclasses.fields(self._norm_stats)
                shapes = {
                    f.name: getattr(getattr(self._norm_stats, f.name), "shape", None)
                    for f in fields
                }
                print(f"[SimAdaptorInference] Norm stats loaded (dataclass {type(self._norm_stats).__name__}): shapes={shapes}")
            else:
                print(f"[SimAdaptorInference] Norm stats loaded (type {type(self._norm_stats)}), no structured keys available")
            ablation_mode = str(getattr(cfg, "ablation_mode", "tam") or "tam")
            if ablation_mode != "tam":
                raise ValueError(
                    "Public TAM inference supports only checkpoints with cfg.ablation_mode='tam'. "
                    f"Got {ablation_mode!r}."
                )
            adaptor = models.SimAdaptorJointwiseFlat(
                emb_dim=cfg.emb_dim,
                hidden=cfg.adaptor_hidden,
                depth=cfg.adaptor_depth,
            )
            self._simadaptor_params = simadaptor_params

        # Resolve XML path from explicit argument or checkpoint metadata only.
        xml_path = self._resolve_xml_path(
            xml_path,
            self._ckpt_metadata_path if self._ckpt_metadata_path is not None else self._ckpt_path,
            self._cfg,
        )

        self._xml_path = str(xml_path)
        print(f"[SimAdaptorInference] Using XML path: {self._xml_path}")
        # Base model (no self-collision) kept for backward compatibility.
        self._mjx_model_template = dynamics.load_mjx_model_from_path(str(xml_path), remove_constraints=True)
        self._drop_incompatible_norm_stats_for_model()

        if self._load_collision_models_requested:
            self._ensure_collision_models_loaded()

        if adaptor is not None and self._cfg is not None:
            data_cfg = getattr(self._cfg, "data", None)
            if not bool(getattr(data_cfg, "ideal_model_has_gravity", True)):
                raise ValueError(
                    "Public TAM inference supports only real-gravity checkpoints "
                    "(legacy cfg.data.ideal_model_has_gravity must be absent or True)."
                )
            self._ideal_model_has_gravity = True
            self._mjx_model_template = self._mjx_model_template.replace(
                body_gravcomp=jnp.zeros_like(self._mjx_model_template.body_gravcomp)
            )
            print("[SimAdaptorInference] Using public TAM real-gravity ideal model.")
            ablation_mode = str(getattr(self._cfg, "ablation_mode", "tam") or "tam")
            if ablation_mode != "tam":
                raise ValueError(
                    "Public TAM inference supports only checkpoints with cfg.ablation_mode='tam'. "
                    f"Got {ablation_mode!r}."
                )
            hist_enc = models_transformer.JointwiseFlatARTransformerDecoder(
                cfg=self._cfg.enc,
                emb_dim=self._cfg.emb_dim,
                ideal_mjx_model=self._mjx_model_template,
            )

            print(
                "Using public TAM jointwise residual head and autoregressive encoder "
                f"(ablation_mode='{ablation_mode}')."
            )
            self._simadaptor_model = (hist_enc, adaptor)

    def _ensure_collision_models_loaded(self) -> None:
        if self._collision_models_loaded:
            return
        xml_path = Path(self._xml_path)
        # Collision-aware model (self + ground) for trajectory vetting.
        # Prefer sibling *_selfcol.xml when available; otherwise use the same XML.
        selfcol_candidate = xml_path.with_name(f"{xml_path.stem}_selfcol.xml")
        if selfcol_candidate.exists():
            self._collision_model_path = str(selfcol_candidate)
        else:
            self._collision_model_path = str(xml_path)
            print(
                f"[SimAdaptorInference] Self-collision XML not found at {selfcol_candidate}; "
                f"using {self._collision_model_path} for collision checks."
            )

        self._mj_model_selfcol = mujoco.MjModel.from_xml_path(self._collision_model_path)
        self._mjx_model_selfcol = mjx.put_model(self._mj_model_selfcol)
        self._arm_joint_ids = rollout.guess_arm_joint_ids(self._mj_model_selfcol, dof_target=7)
        self._arm_qpos_idx = np.array([self._mj_model_selfcol.jnt_qposadr[j] for j in self._arm_joint_ids], dtype=np.int32)
        self._arm_joint_range = np.array(self._mj_model_selfcol.jnt_range[self._arm_joint_ids])
        self._clearance_models = {}  # margin -> (model, arm_qpos_idx)
        self._collision_models_loaded = True

    def _norm_stats_dof(self) -> int | None:
        stats = getattr(self, "_norm_stats", None)
        if stats is None:
            return None
        for key in ("mean_q", "var_q"):
            if isinstance(stats, Mapping):
                value = stats.get(key)
            else:
                value = getattr(stats, key, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and int(arr.shape[-1]) > 0:
                return int(arr.shape[-1])
        return None

    def _drop_incompatible_norm_stats_for_model(self) -> None:
        stats_dof = self._norm_stats_dof()
        if stats_dof is None:
            return
        model_dof = int(getattr(self._mjx_model_template, "nu", 0) or 0)
        if model_dof <= 0 or stats_dof == model_dof:
            return
        print(
            "[SimAdaptorInference] Dropping checkpoint norm stats because their "
            f"DoF ({stats_dof}) does not match the target model DoF ({model_dof}). "
            "This is expected for cross-DoF jointwise checkpoint evals."
        )
        self._norm_stats = None

    def _infer_export_dof(self, params: dict, cfg: object) -> int:
        arm_joint_ids = getattr(self, "_arm_joint_ids", None)
        if arm_joint_ids is not None:
            try:
                dof = int(len(arm_joint_ids))
            except Exception:
                dof = 0
            if dof > 0:
                return dof

        norm_dof = self._norm_stats_dof()
        if norm_dof is not None:
            return norm_dof

        model = getattr(self, "_mjx_model_template", None)
        for attr in ("nu", "nv", "nq"):
            value = int(getattr(model, attr, 0) or 0)
            if value > 0:
                return min(value, 7)
        raise RuntimeError(
            "Failed to infer SimAdaptor export DoF without collision models. "
            "Pass load_collision_models=True or provide norm stats / model dimensions."
        )

    @property
    def ckpt_path(self) -> Path | None:
        return self._ckpt_path

    @property
    def cfg(self):
        return self._cfg

    @property
    def checkpoint_metadata(self) -> dict:
        """Return the run-level checkpoint metadata loaded from ``save_dict.pkl``."""

        return dict(self._checkpoint_metadata_payload)

    @property
    def dagger_cfg(self):
        """Return online-DAgger metadata when the checkpoint provides it."""

        return self._checkpoint_metadata_payload.get("dagger_cfg")

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mjx_model_template(self) -> mjx.Model:
        return self._mjx_model_template

    @property
    def adaptor_seq_length(self) -> int:
        if self._cfg is None:
            return 1
        return max(int(getattr(self._cfg, "adaptor_seq_length", 1) or 1), 1)

    @property
    def ideal_model_has_gravity(self) -> bool:
        return True

    def _ensure_checkpoint_loaded(self):
        if self._simadaptor_model is None or self._simadaptor_params is None:
            raise RuntimeError("SimAdaptor checkpoint required. Provide a checkpoint path when constructing SimAdaptorInference.")

    def get_history_runtime_bundle(self) -> HistoryRuntimeBundle:
        self._ensure_checkpoint_loaded()
        return HistoryRuntimeBundle(
            hist_model=self._simadaptor_model[0],
            hist_params=self._simadaptor_params["hist"],
            norm_stats=self._norm_stats,
            mjx_model_template=self._mjx_model_template,
            ideal_model_has_gravity=self.ideal_model_has_gravity,
        )

    def get_apply_fns(self):
        """Return JIT'd (history_apply_fn, adaptor_apply_fn, params, norm_stats, cfg).

        Signatures match the training script wrappers.
        """
        self._ensure_checkpoint_loaded()

        if getattr(self, "_history_apply_jit", None) is None or getattr(self, "_adaptor_apply_jit", None) is None:
            hist_enc, adaptor = self._simadaptor_model
            self._history_apply_jit = jax.jit(
                lambda params_hist, q, qd, u, rng, deterministic, norm_stats: hist_enc.apply(
                    params_hist,
                    q,
                    qd,
                    u,
                    rngs={"dropout": rng},
                    deterministic=deterministic,
                    norm_stats=norm_stats,
                ),
                static_argnums=5,  # deterministic
            )
            self._adaptor_apply_jit = jax.jit(
                lambda params_adaptor, q, qd, tau, hist, rng, train, norm_stats: adaptor.apply(
                    params_adaptor,
                    q,
                    qd,
                    tau,
                    hist,
                    train=train,
                    rngs={"dropout": rng},
                    norm_stats=norm_stats,
                ),
                static_argnums=6,  # train
            )

        return self._history_apply_jit, self._adaptor_apply_jit, self._simadaptor_params, self._norm_stats, self._cfg
    
    def gen_activation_trajectory(self, key, mjx_model:mjx.Model, mjx_params):
        # mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(
        #                                     disableflags=mjx_model.opt.disableflags | mjx.DisableBit.CONTACT,
        #                                     iterations=1, ls_iterations=1, timestep=0.001, integrator=mujoco.mjtIntegrator.mjINT_EULER))
        mjx_model_ptb = self._mjx_model_template
        mjx_model_ptb = mjx_model_ptb.replace(opt=mjx_model_ptb.opt.replace(
                                            disableflags=mjx_model_ptb.opt.disableflags | mjx.DisableBit.CONTACT,
                                            iterations=1, ls_iterations=1, timestep=0.001, integrator=mujoco.mjtIntegrator.mjINT_EULER))
        omit = {"actuator_ctrlrange", "actuator_forcerange", "jnt_actfrcrange", "actuator_gear"}
        common_nu = min(mjx_model.nu, mjx_model_ptb.nu)
        limit_params = {k:v[...,:common_nu,:] for k,v in mjx_params.items() if k in omit}
        mjx_model_ptb = mjx_model_ptb.tree_replace(limit_params)
        Kp = jnp.array([80, 70, 70, 50, 20, 20, 10], dtype=jnp.float32)*0.3
        Kd = 2.0*jnp.sqrt(Kp)  # critical damping
        actuator_params = {
            'kp': Kp,
            'kd': Kd,
            'torque_range': mjx_model_ptb.actuator_forcerange,
        }
        # Keep only rollout-parameter fields; env/model patch dicts may also contain
        # render/setup fields such as body_pos or geom_pos.
        rollout_param_keys = set(structs.RolloutParams.__annotations__.keys()) - set(actuator_params.keys())
        mjx_params_in = {
            k: v for k, v in mjx_params.items() if k not in omit and k in rollout_param_keys
        }
        rollout_params = structs.RolloutParams(**mjx_params_in, **actuator_params)
        rollout_params = rollout_params.fit_model_size(mjx_model_ptb)[None]
        dof_idx_arm = jnp.arange(7)
        activation_rollouts = rollout.rollout_generation(
            key,
            rollout_params.controller_params.get_actuator_fn(
                control_type='qref',
                ideal_mjx_model=mjx_model_ptb,
            ),
            mjx_model_ptb,
            dof_idx_arm,
            rollout_params,
            pause_prob=0.0,
            num_waypoints=5,
            duration=8.0,
        )
        activation_rollouts = jax.tree_util.tree_map(lambda x: x[0], activation_rollouts)
        return activation_rollouts
        
    def history_encoding(
        self,
        q_hist,
        dq_hist,
        u_hist,
        rng,
        times=None,
        input_keep_mask=None,
        *,
        gravity=None,
        tau_is_model_space: bool = False,
    ):
        self._ensure_checkpoint_loaded()
        # Backward-compatible behavior: when gravity is not supplied, treat the
        # incoming torque history as already prepared for the history encoder.
        tau_is_model_space = bool(tau_is_model_space or gravity is None)
        q_hist, dq_hist, u_hist, _ = prepare_history_inputs(
            q_hist,
            dq_hist,
            u_hist,
            gravity=gravity,
            ideal_model_has_gravity=self.ideal_model_has_gravity,
            context="SimAdaptorInference.history_encoding",
            tau_is_model_space=tau_is_model_space,
            apply_zero_torque_mask=False,
        )
        extended = False
        if q_hist.ndim == 2:
            extended = True
            q_hist = q_hist[None, ...]
            dq_hist = dq_hist[None, ...]
            u_hist = u_hist[None, ...]
        if times is not None:
            q_hist, dq_hist, _, u_hist, t_uniform = smoothing_util.q_traj_to_traj(q_hist, u_hist, times)
        history_emb = self._simadaptor_model[0].apply(
            self._simadaptor_params['hist'],
            q_hist,
            dq_hist,
            u_hist,
            deterministic=True,
            rngs={'dropout': rng},
            norm_stats=self._norm_stats,
            input_keep_mask=input_keep_mask,
        )
        if extended:
            history_emb = history_emb[0]
        return history_emb

    def online_history_update(self, q_seq, dq_seq, u_seq, cache=None, valid_mask=None, rng=None):
        self._ensure_checkpoint_loaded()
        warnings.warn(
            "SimAdaptorInference.online_history_update() is deprecated. "
            "Prefer RealTimeHistoryAdaptor or simadaptor.models.transformer.online_history_update().",
            DeprecationWarning,
            stacklevel=2,
        )
        """
        Autoregressively update the history embedding using the decoder's cache.

        Args:
            q_seq/dq_seq/u_seq: token inputs shaped [B, T, patch, 7] or [T, patch, 7].
                                 Missing batch/time dims will be added automatically.
            cache: decode cache from a previous call (or None to initialize).
            valid_mask: optional mask [B, T] marking valid tokens.
            rng: optional PRNGKey for dropout.
        Returns:
            history_emb: [B, T, emb_dim] embeddings for each token.
            cache: updated decode cache.
        """
        return models_transformer.online_history_update(
            params=self._simadaptor_params['hist'],
            model=self._simadaptor_model[0],
            q_seq=q_seq,
            qd_seq=dq_seq,
            u_seq=u_seq,
            cache=cache,
            valid_mask=valid_mask,
            key=rng,
            norm_stats=self._norm_stats,
        )
    
    def adaptor(self, q_window, dq_window, u_des_window, history_emb):
        self._ensure_checkpoint_loaded()
        # Accept either [T, DoF] or [B, T, DoF] for q/dq/u_des.
        if q_window.shape != dq_window.shape or q_window.shape != u_des_window.shape:
            raise ValueError(f"q, dq, u_des must share shape; got {q_window.shape}, {dq_window.shape}, {u_des_window.shape}")
        extended = False
        if q_window.ndim == 2:
            extended = True
            q_window = q_window[None, ...]
            dq_window = dq_window[None, ...]
            u_des_window = u_des_window[None, ...]
        elif q_window.ndim != 3:
            raise ValueError(f"q, dq, u_des must have rank 2 or 3; got {q_window.shape}")
        if history_emb.ndim == 1:
            history_emb = history_emb[None, ...]
        elif (
            history_emb.ndim == 2
            and history_emb.shape[0] != q_window.shape[0]
            and history_emb.shape[0] == q_window.shape[-1]
        ):
            history_emb = history_emb[None, ...]
        delta_tau, _ = self._simadaptor_model[1].apply(
            self._simadaptor_params['adaptor'],
            q_window,
            dq_window,
            u_des_window,
            history_emb,
            norm_stats=self._norm_stats,
        )
        
        # dead zone scaling (optional)
        # delta_tau = adaptor_deadzone_scaling(u_des_window[...,-1,:], delta_tau)
        tau_res = u_des_window[...,-1,:] + delta_tau
        if extended:
            tau_res = tau_res[0]
        return tau_res
        # tau_res = actuator_util.calculate_gt_torque(history_emb, q, dq, mjx_model_ideal=self._mjx_model_template, tau=u_des)
        # return tau_res

    def _has_collision(self, q_traj: np.ndarray, stride: int = 10) -> bool:
        """
        Cheap collision check along a trajectory using the collision model (robot + ground plane).
        """
        self._ensure_collision_models_loaded()
        data = mujoco.MjData(self._mj_model_selfcol)
        for q in q_traj[::stride]:
            data.qpos[:] = self._mj_model_selfcol.qpos0
            data.qpos[self._arm_qpos_idx] = q
            mujoco.mj_forward(self._mj_model_selfcol, data)
            if data.ncon > 0:
                return True
        return False

    def _get_clearance_model(self, margin: float):
        """
        Return a cached model with inflated geom margins to measure minimum clearance.
        """
        self._ensure_collision_models_loaded()
        if margin not in self._clearance_models:
            model = mujoco.MjModel.from_xml_path(self._collision_model_path)
            model.geom_margin[:] = margin
            arm_qpos_idx = np.array([model.jnt_qposadr[j] for j in self._arm_joint_ids], dtype=np.int32)
            self._clearance_models[margin] = (model, arm_qpos_idx)
        return self._clearance_models[margin]

    def _min_clearance(self,
                       q_traj: np.ndarray,
                       margin: float = 0.003,
                       stride: int = 10,
                       num_perturbations: int = 0,
                       noise_std: float = 0.02):
        """
        Compute minimum signed contact distance along a trajectory (and optional perturbations).
        Positive values mean separation; negative means penetration. Uses inflated margins to
        approximate clearance.
        """
        model, arm_qpos_idx = self._get_clearance_model(margin)
        data = mujoco.MjData(model)
        min_dist = np.inf
        collided = False

        def maybe_update(q_vec):
            nonlocal min_dist, collided
            data.qpos[:] = model.qpos0
            data.qpos[arm_qpos_idx] = q_vec
            mujoco.mj_forward(model, data)
            if data.ncon > 0:
                collided = True
                for c in range(data.ncon):
                    min_dist = min(min_dist, float(data.contact[c].dist))
                    contact = data.contact[c]
                    g1, g2 = contact.geom1, contact.geom2
                    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

                    b1 = model.geom_bodyid[g1]
                    b2 = model.geom_bodyid[g2]
                    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
                    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)

                    print(f"{geom1_name} (body {body1_name}) vs {geom2_name} (body {body2_name}), dist={contact.dist}")

        for q in q_traj[::stride]:
            maybe_update(q)
            if collided:
                return collided, min_dist

        if num_perturbations > 0:
            key = jax.random.PRNGKey(0)  # deterministic since for analysis
            for i in range(num_perturbations):
                key, sub = jax.random.split(key)
                noise = jax.random.normal(sub, q_traj.shape[1:]) * noise_std
                q_noisy = np.clip(q_traj[::stride] + np.array(noise), self._arm_joint_range[:, 0], self._arm_joint_range[:, 1])
                for q in q_noisy:
                    maybe_update(q)

        if not collided:
            min_dist = np.inf if np.isinf(min_dist) else min_dist

        return collided, min_dist

    def _violates_joint_limits(self, q_traj: np.ndarray, tol: float = 1e-3) -> bool:
        self._ensure_collision_models_loaded()
        low = self._arm_joint_range[:, 0] - tol
        high = self._arm_joint_range[:, 1] + tol
        below = (q_traj < low).any()
        above = (q_traj > high).any()
        return bool(below or above)

    def generate_collision_free_reference(self,
                                          key,
                                          num_waypoints: int = 5,
                                          duration: float = 8.0,
                                          dt: float = 0.002,
                                          max_tries: int = 1024,
                                          waypoint_margin: float = 0.1,
                                          collision_stride: int = 10,
                                          min_clearance: float = 0.05,
                                          clearance_margin: float = 0.003,
                                          clearance_stride: int = 10,
                                          clearance_num_perturb: int = 3,
                                          clearance_noise_std: float = 0.02,
                                          workspace_box=None,
                                          workspace_site: str = "gripper",
                                          max_joint_vel: float | None = None):
        """
        Sample random waypoints and build an interpolated reference that is
        collision free under the configured collision model (self-collision + ground plane when available).
        Also checks the minimum
        contact distance with an inflated margin for extra safety.

        Returns a dict containing waypoints, q_ref/qd_ref, times, and rollout_params.
        """
        self._ensure_collision_models_loaded()
        joint_l = jnp.array(self._arm_joint_range[:, 0])
        joint_h = jnp.array(self._arm_joint_range[:, 1])
        span = joint_h - joint_l
        joint_l = joint_l + waypoint_margin * span
        joint_h = joint_h - waypoint_margin * span
        
        # Standard Franka Emika Panda "home" / neutral joint pose (radians)
        home_position = jnp.array([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483], dtype=jnp.float32)

        key_sample = key
        q_ref = None
        qd_ref = None
        waypoints = None
        if workspace_box is None:
            # Default workspace box in world frame (meters): (min, max)
            workspace_box = (
                np.array([0.25, -0.4, -0.05], dtype=float),
                np.array([0.85, 0.4, 0.6], dtype=float),
            )
        workspace_box = tuple(np.asarray(b, dtype=float) for b in workspace_box)
        joint_span = joint_h - joint_l

        # Precompute FK for workspace rejection (use gripper site).
        mj_model = self._mj_model_selfcol
        site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, workspace_site)
        workspace_check_enabled = site_id != -1
        if not workspace_check_enabled:
            print(f"[generate_collision_free_reference] Workspace site '{workspace_site}' not found; skipping workspace check.")
        data = mujoco.MjData(mj_model)

        def _build_traj_with_zero_end_vel(wps, duration, dt):
            """Cubic interpolate waypoints with zero velocity at start/end; fallback to default builder."""
            try:
                from scipy.interpolate import CubicSpline
            except Exception:
                return rollout.build_traj_from_waypoints(wps, int(duration / dt), duration)
            wps_np = np.asarray(wps)
            knot_times = np.linspace(0.0, duration, wps_np.shape[0])
            T = int(duration / dt)
            target_t = np.linspace(0.0, duration, T, endpoint=False)
            q_list, dq_list = [], []
            for j in range(wps_np.shape[1]):
                cs = CubicSpline(knot_times, wps_np[:, j], bc_type=((1, 0.0), (1, 0.0)))
                q_list.append(cs(target_t))
                dq_list.append(cs(target_t, 1))
            q_ref_np = np.stack(q_list, axis=1)
            qd_ref_np = np.stack(dq_list, axis=1)
            return jnp.asarray(q_ref_np), jnp.asarray(qd_ref_np)

        for itr in tqdm.tqdm(range(max_tries)):
            key_sample, sub = jax.random.split(key_sample)
            sub_keys = jax.random.split(sub, num_waypoints)
            waypoints = rollout.generate_waypoints(key_sample, num_waypoints, batch_n=1, dof=7, 
                                           joint_range=jnp.stack([joint_l, joint_h], axis=-1), pause_prob=0.0,
                                           initial_wp=home_position)
            waypoints = waypoints[0]
            # waypoints = [home_position]
            # for i in range(1, num_waypoints):
            #     prev = waypoints[-1]
            #     # sample delta in a band: at least half span away, but clamp to limits
            #     delta_mag = jnp.deg2rad(100)
            #     candidate = jax.random.uniform(sub_keys[i], shape=prev.shape, minval=(prev-delta_mag).clip(joint_l, joint_h), maxval=(prev+delta_mag).clip(joint_l, joint_h))
            #     # candidate = jnp.clip(candidate, joint_l, joint_h)
            #     waypoints.append(candidate)
            # waypoints = jnp.stack(waypoints, axis=0)
            q_ref, qd_ref = _build_traj_with_zero_end_vel(waypoints, duration, dt)
            if self._violates_joint_limits(np.asarray(q_ref)):
                continue
            if max_joint_vel is not None:
                if jnp.any(jnp.abs(qd_ref) > max_joint_vel):
                    continue
            if self._has_collision(np.asarray(q_ref), stride=collision_stride):
                continue
            # Workspace check: ensure gripper site stays inside box.
            if workspace_check_enabled:
                data.qpos[:] = mj_model.qpos0
                inside = True
                for q in np.asarray(q_ref)[::collision_stride]:
                    data.qpos[self._arm_qpos_idx] = q
                    mujoco.mj_forward(mj_model, data)
                    pos = np.asarray(data.site_xpos[site_id])
                    if np.any(pos < workspace_box[0]) or np.any(pos > workspace_box[1]):
                        inside = False
                        break
                if not inside:
                    continue
            collided, min_dist = self._min_clearance(
                np.asarray(q_ref),
                margin=clearance_margin,
                stride=clearance_stride,
                num_perturbations=clearance_num_perturb,
                noise_std=clearance_noise_std,
            )
            if collided or (min_dist < min_clearance):
                continue
            break
        else:
            raise RuntimeError("Failed to sample collision-free trajectory after max_tries.")

        times = jnp.linspace(0.0, duration, q_ref.shape[0], endpoint=False)
        
        return q_ref, qd_ref, times, min_dist

    
    # ---- C++ export helpers ----
    @staticmethod
    def _write_dense(dense_params: dict, f, use_bias: bool = True):
        np.asarray(dense_params["kernel"], dtype=np.float32).tofile(f)
        if use_bias and "bias" in dense_params:
            np.asarray(dense_params["bias"], dtype=np.float32).tofile(f)

    @staticmethod
    def _write_layernorm(ln_params: dict, f):
        np.asarray(ln_params["scale"], dtype=np.float32).tofile(f)
        np.asarray(ln_params["bias"], dtype=np.float32).tofile(f)

    def _write_block(self, block_params: dict, f):
        # AdaLN 1
        self._write_layernorm(block_params["AdaLN_0"]["LayerNorm_0"], f)
        self._write_dense(block_params["AdaLN_0"]["Dense_0"], f, use_bias=True)
        # Dense 0
        self._write_dense(block_params["Dense_0"], f, use_bias=True)
        # AdaLN 2
        self._write_layernorm(block_params["AdaLN_1"]["LayerNorm_0"], f)
        self._write_dense(block_params["AdaLN_1"]["Dense_0"], f, use_bias=True)
        # Dense 1
        self._write_dense(block_params["Dense_1"], f, use_bias=True)

    def export_simadaptor_weights_cpp(
        self,
        out_path: str | Path,
    ):
        """
        Export adaptor weights to the controller-side C++ SimAdaptor binary format.
        """
        self._ensure_checkpoint_loaded()
        params = self._simadaptor_params["adaptor"]
        # Some checkpoints store an extra "params" nesting.
        if "q_stem" not in params and "params" in params:
            params = params["params"]
        cfg = self._cfg
        dof = self._infer_export_dof(params, cfg)
        history = cfg.adaptor_seq_length
        legacy_direct_head_names = (
            "joint_direct_tau_hyper",
            "joint_direct_projected",
            "joint_direct_projected_out",
        )
        command_head_names = (
            "joint_direct_tau_hyper",
            "joint_direct_projected2",
        )
        has_jointwise_command_head = (
            all(name in params for name in command_head_names)
            and "joint_direct_projected_out" in params
        )
        has_legacy_jointwise_direct_head = all(
            name in params for name in legacy_direct_head_names
        )
        use_jointwise_direct_head = has_jointwise_command_head or has_legacy_jointwise_direct_head
        if (
            not use_jointwise_direct_head
            and "joint_out" not in params
        ):
            raise NotImplementedError(
                "C++ adaptor export for jointwise no-command-map checkpoints requires "
                "either the legacy joint_out block or the direct-residual head params "
                f"{legacy_direct_head_names} / joint_direct_projected2; "
                f"got keys={sorted(params.keys())}."
            )
        out_path = Path(out_path)
        with out_path.open("wb") as f:
            flags = 0
            flags |= _ADAPTOR_FLAG_JOINTWISE_CONDITIONING
            if use_jointwise_direct_head:
                flags |= _ADAPTOR_FLAG_JOINTWISE_DIRECT_RESIDUAL_HEAD
            if has_jointwise_command_head:
                flags |= _ADAPTOR_FLAG_COMMAND_CONDITIONED_HEAD
            np.array(
                [
                    dof,
                    cfg.emb_dim,
                    cfg.adaptor_hidden,
                    cfg.adaptor_depth,
                    cfg.adaptor_seq_length,
                    flags,
                ],
                dtype=np.int32,
            ).tofile(f)
            self._write_dense(params["q_stem"], f, use_bias=False)
            self._write_dense(params["qd_stem"], f, use_bias=False)
            self._write_dense(params["tau_stem"], f, use_bias=False)
            self._write_layernorm(params["q_stem_ln"], f)
            self._write_layernorm(params["qd_stem_ln"], f)
            self._write_layernorm(params["tau_stem_ln"], f)
            for i in range(cfg.adaptor_depth - 1):
                self._write_block(params[f"SimAdaptorBlock_{i}"], f)
                self._write_dense(params[f"joint_global_proj_{i}"], f, use_bias=True)
            if has_jointwise_command_head:
                self._write_dense(params["joint_direct_tau_hyper"], f, use_bias=True)
                self._write_dense(params["joint_direct_projected2"], f, use_bias=True)
                self._write_dense(params["joint_direct_projected_out"], f, use_bias=True)
            elif has_legacy_jointwise_direct_head:
                self._write_dense(params["joint_direct_tau_hyper"], f, use_bias=True)
                self._write_dense(params["joint_direct_projected"], f, use_bias=True)
                self._write_dense(params["joint_direct_projected_out"], f, use_bias=True)
            else:
                self._write_block(params["joint_out"], f)
            if self._norm_stats is not None:
                packed = _pack_norm_stats(self._norm_stats, dof, history)
                packed.tofile(f)
        return out_path

def _pack_norm_stats(norm_stats, dof: int, history: int) -> np.ndarray:
    """Accepts a dict/obj with mean_q, mean_dq/mean_qd, mean_u/mean_tau, var_q, var_dq/var_qd, var_u/var_tau."""
    def _get(keys):
        if isinstance(norm_stats, Mapping):
            for key in keys:
                if key in norm_stats and norm_stats[key] is not None:
                    return np.asarray(norm_stats[key], dtype=np.float32)
        else:
            for key in keys:
                value = getattr(norm_stats, key, None)
                if value is not None:
                    return np.asarray(value, dtype=np.float32)
        raise KeyError(f"Missing norm_stats field; tried {keys}")

    mean_q = _get(("mean_q",))
    mean_dq = _get(("mean_dq", "mean_qd"))
    mean_tau = _get(("mean_u", "mean_tau"))
    var_q = _get(("var_q",))
    var_dq = _get(("var_dq", "var_qd"))
    var_tau = _get(("var_u", "var_tau"))
    
    # print mean q and var q for debugging
    print("Norm stats:")
    print(" mean_q:", mean_q)
    print(" var_q:", var_q)
    return np.concatenate([mean_q, mean_dq, mean_tau, var_q, var_dq, var_tau]).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(
        "Use tam-mapping-server or tam-eval-source-to-osc-sim instead of running "
        "simadaptor.deploy.inf_util directly."
    )
    
