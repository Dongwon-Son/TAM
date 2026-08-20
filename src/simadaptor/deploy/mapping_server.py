from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
import zmq

from simadaptor.deploy.history_client import (
    COMMAND_ENDPOINT,
    HISTORY_ENDPOINT,
    REQUEST_ENDPOINT,
    HistoryControllerClient,
)
from simadaptor.deploy.mapping_server_meta import (
    DEFAULT_MAPPING_CONTROL_ENDPOINT,
    MAPPING_MODE_NONE,
    MAPPING_MODE_SIMADAPTOR,
)
from simadaptor.deploy.runtime_common import (
    extract_history_window_arrays,
    flatten_history_embedding_for_transport,
)

try:
    import select
    import termios
    import tty
except Exception:  # pragma: no cover
    select = None
    termios = None
    tty = None


DEFAULT_DEPLOY_ATTENTION_HISTORY_S = 4.0
HISTORY_TORQUE_MODE_AUTO = "auto"
HISTORY_TORQUE_MODE_APPLIED = "applied"
HISTORY_TORQUE_MODE_BASE_TAM_FUSION = "base_tam_fusion"
HISTORY_TORQUE_MODE_CHOICES = (
    HISTORY_TORQUE_MODE_AUTO,
    HISTORY_TORQUE_MODE_APPLIED,
    HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
)
_APPLIED_TAU_KEYS = ("tau_applied", "tau_cmd", "tau_commanded", "tau", "u_des", "u", "tau_measured")
_BASE_TAU_KEYS = (
    "tau_base",
    "tau_base_cmd",
    "base_tau",
    "base_tau_cmd",
    "tau_plain",
    "tau_nominal",
    "tau_policy",
    "tau_without_adaptor",
)
_TAM_RESIDUAL_TAU_KEYS = (
    "tau_tam_residual",
    "tam_residual",
    "tau_adaptor_delta",
    "tau_delta",
)
_REMOTE_PREPARE_RETRY_INTERVAL_S = 0.5
_HOLD_RESEND_INTERVAL_S = 0.2


class AdaptorBinUploadError(RuntimeError):
    """Raised when the controller explicitly rejects an uploaded adaptor bin."""


def _normalize_backend_name(backend: Any) -> str:
    token = str(backend or "").strip().lower().replace("-", "_")
    if token in {"tam", "simadaptor", "sim_adaptor"}:
        return "tam"
    return token


def _mapping_mode_for_backend(backend: Any) -> str:
    token = _normalize_backend_name(backend)
    if token == "tam":
        return MAPPING_MODE_SIMADAPTOR
    return MAPPING_MODE_NONE


def _simadaptor_ablation_mode(adaptor: Any) -> str:
    inf = getattr(adaptor, "inf", None)
    cfg = getattr(inf, "cfg", None)
    return str(getattr(cfg, "ablation_mode", "tam") or "tam").strip() or "tam"


def _simadaptor_config_history_torque_mode(adaptor: Any) -> str:
    inf = getattr(adaptor, "inf", None)
    dagger_cfg = getattr(inf, "dagger_cfg", None)
    dagger_mode = getattr(dagger_cfg, "history_torque_mode", None)
    if dagger_mode:
        return str(dagger_mode).strip()
    cfg = getattr(inf, "cfg", None)
    cfg_mode = getattr(cfg, "history_torque_mode", None)
    if cfg_mode:
        return str(cfg_mode).strip()
    if _simadaptor_has_history_fusion(adaptor):
        return HISTORY_TORQUE_MODE_BASE_TAM_FUSION
    return HISTORY_TORQUE_MODE_APPLIED


def _simadaptor_dagger_config(adaptor: Any) -> Any:
    inf = getattr(adaptor, "inf", None)
    return getattr(inf, "dagger_cfg", None)


def _simadaptor_has_history_fusion(adaptor: Any) -> bool:
    inf = getattr(adaptor, "inf", None)
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        return "history_fusion" in params
    except Exception:
        return False


def _simadaptor_history_fusion_params(adaptor: Any) -> Any:
    inf = getattr(adaptor, "inf", None)
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        return params["history_fusion"]
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint requested base_tam_fusion history, but history_fusion "
            "parameters are missing."
        ) from exc


def _resolve_simadaptor_history_torque_mode(adaptor: Any, requested: str) -> str:
    requested = str(requested or HISTORY_TORQUE_MODE_AUTO).strip()
    if requested not in HISTORY_TORQUE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported history torque mode {requested!r}; "
            f"choose one of {HISTORY_TORQUE_MODE_CHOICES}."
        )
    cfg_mode = _simadaptor_config_history_torque_mode(adaptor)
    mode = cfg_mode if requested == HISTORY_TORQUE_MODE_AUTO else requested
    if mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
        if not _simadaptor_has_history_fusion(adaptor):
            raise RuntimeError(
                "base_tam_fusion history requires checkpoint params['history_fusion']; "
                f"checkpoint history_torque_mode={cfg_mode!r} but no fusion weights were found."
            )
        return HISTORY_TORQUE_MODE_BASE_TAM_FUSION
    return HISTORY_TORQUE_MODE_APPLIED


def _apply_history_fusion(
    fusion_params: Any,
    history_emb_applied: Any,
    history_emb_base: Any,
    history_emb_tam: Any,
) -> Any:
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


def _format_age_s(age_s: Optional[float]) -> str:
    if age_s is None:
        return "never"
    age_s = max(0.0, float(age_s))
    if age_s < 1.0:
        return f"{age_s:.2f}s"
    if age_s < 60.0:
        return f"{age_s:.1f}s"
    return f"{age_s / 60.0:.1f}m"


def _status_age(now: float, last_event_wall: Optional[float]) -> str:
    if last_event_wall is None:
        return "never"
    return _format_age_s(now - float(last_event_wall))


def _age_s(now: float, last_event_wall: Optional[float]) -> Optional[float]:
    if last_event_wall is None:
        return None
    return max(0.0, float(now) - float(last_event_wall))


def _heartbeat_text(interval_s: float) -> str:
    return "disabled" if float(interval_s) <= 0.0 else f"every {_format_age_s(interval_s)}"


def _format_array_preview(
    value: Any,
    *,
    max_items: int = 8,
    precision: int = 4,
) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    preview = arr[: max(1, int(max_items))]
    suffix = "" if arr.size <= preview.size else f", ... ({arr.size} total)"
    return (
        np.array2string(
            preview,
            precision=int(precision),
            suppress_small=True,
            separator=", ",
            max_line_width=200,
        )
        + suffix
    )


def _describe_embedding_for_log(history_emb: np.ndarray) -> str:
    arr = np.asarray(history_emb, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "embedding=[]"
    return (
        f"dim={arr.size} "
        f"min={float(np.min(arr)):.5g} "
        f"max={float(np.max(arr)):.5g} "
        f"preview={_format_array_preview(arr)}"
    )


def _cfg_value(obj: Any, name: str, default: Any = "n/a") -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _format_cfg_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_format_cfg_value(v) for v in value) + ")"
    return str(value)


def _format_cfg_items(**items: Any) -> str:
    return " ".join(
        f"{key}={_format_cfg_value(value)}" for key, value in items.items()
    )


def _adaptor_param_keys(inf: Any) -> set[str]:
    params = getattr(inf, "_simadaptor_params", {}) or {}
    adaptor_params = params.get("adaptor", {}) if isinstance(params, dict) else {}
    if isinstance(adaptor_params, dict) and "params" in adaptor_params:
        adaptor_params = adaptor_params["params"]
    if not isinstance(adaptor_params, dict):
        return set()
    return set(str(key) for key in adaptor_params.keys())


def _describe_simadaptor_head(cfg: Any, inf: Any) -> str:
    del cfg
    keys = _adaptor_param_keys(inf)
    has_direct = {
        "joint_direct_tau_hyper",
        "joint_direct_projected",
        "joint_direct_projected_out",
    }.issubset(keys)
    if has_direct:
        return "jointwise_direct_residual(joint_direct_*)"
    return "jointwise_legacy_residual(joint_out)"


def _jsonable_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        return None
    return None if text == "" or text == "n/a" else text


def _simadaptor_checkpoint_meta(
    *,
    args: argparse.Namespace,
    adaptor: Any,
    bin_name: str | None,
    bin_b64: str | None,
) -> dict[str, Any]:
    inf = adaptor.inf
    cfg = getattr(inf, "cfg", None)
    dagger_cfg = _simadaptor_dagger_config(adaptor)
    ckpt_cfg = _cfg_value(cfg, "ckpt", None)
    data = _cfg_value(cfg, "data", None)

    if args.ckpt_path is not None:
        source = "ckpt_path"
        source_id = _jsonable_path(args.ckpt_path)
    else:
        source = "unknown"
        source_id = None

    return {
        "source": source,
        "source_id": source_id,
        "resolved_ckpt": _jsonable_path(getattr(inf, "ckpt_path", None)),
        "ckpt_metadata_path": _jsonable_path(getattr(inf, "_ckpt_metadata_path", None)),
        "run_name": _cfg_value(ckpt_cfg, "run_name", None),
        "robot_key": _cfg_value(cfg, "robot_key", None),
        "xml_path": _jsonable_path(getattr(inf, "xml_path", None)),
        "model_head": _describe_simadaptor_head(cfg, inf),
        "ablation_mode": _cfg_value(cfg, "ablation_mode", None),
        "cfg_history_torque_mode": _cfg_value(cfg, "history_torque_mode", None),
        "dagger_history_torque_mode": _cfg_value(
            dagger_cfg,
            "history_torque_mode",
            None,
        ),
        "dagger_attention_history_s": _cfg_value(
            dagger_cfg,
            "attention_history_s",
            None,
        ),
        "resolved_history_torque_mode": getattr(
            args,
            "resolved_history_torque_mode",
            None,
        ),
        "emb_dim": _cfg_value(cfg, "emb_dim", None),
        "ideal_model_contract": "public_tam_real_gravity",
        "attention_history_s": getattr(args, "attention_history_s", None),
        "require_explicit_fused_history": bool(
            getattr(args, "require_explicit_fused_history", True)
        ),
        "bin_name": bin_name,
        "bin_base64_bytes": int(len(bin_b64 or "")),
    }


def _print_simadaptor_deploy_config(
    *,
    args: argparse.Namespace,
    adaptor: Any,
    bin_name: str | None,
    bin_b64: str | None,
) -> None:
    inf = adaptor.inf
    cfg = getattr(inf, "cfg", None)
    dagger_cfg = _simadaptor_dagger_config(adaptor)
    enc = _cfg_value(cfg, "enc", None)
    data = _cfg_value(cfg, "data", None)
    ckpt_cfg = _cfg_value(cfg, "ckpt", None)

    if args.ckpt_path is not None:
        source = "ckpt_path"
        source_id = args.ckpt_path
    else:
        source = "unknown"
        source_id = "n/a"

    print("[mapping_server] TAM deploy config:")
    print(
        "[mapping_server]   checkpoint "
        + _format_cfg_items(
            source=source,
            source_id=source_id,
            resolved_ckpt=getattr(inf, "ckpt_path", None),
            run_name=_cfg_value(ckpt_cfg, "run_name"),
        )
    )
    print(
        "[mapping_server]   model "
        + _format_cfg_items(
            robot_key=_cfg_value(cfg, "robot_key"),
            dof=getattr(adaptor, "_dof", "n/a"),
            tam_head="jointwise_residual",
            head=_describe_simadaptor_head(cfg, inf),
            cfg_history_torque_mode=_cfg_value(cfg, "history_torque_mode"),
            dagger_history_torque_mode=_cfg_value(
                dagger_cfg,
                "history_torque_mode",
            ),
            resolved_history_torque_mode=getattr(
                args,
                "resolved_history_torque_mode",
                "n/a",
            ),
            has_history_fusion="history_fusion" in (getattr(inf, "_simadaptor_params", {}) or {}),
            emb_dim=_cfg_value(cfg, "emb_dim"),
            hidden=_cfg_value(cfg, "adaptor_hidden"),
            depth=_cfg_value(cfg, "adaptor_depth"),
            seq_length=_cfg_value(cfg, "adaptor_seq_length"),
        )
    )
    print(
        "[mapping_server]   encoder "
        + _format_cfg_items(
            mode="autoregressive_rope_jointwise_tam",
            d_model=_cfg_value(enc, "d_model"),
            layers=_cfg_value(enc, "num_layers"),
            heads=_cfg_value(enc, "num_heads"),
            patch_size=_cfg_value(enc, "patch_size"),
            patch_stride=_cfg_value(enc, "patch_stride"),
            torque_residual_feature=True,
            masked_fit_half=_cfg_value(enc, "masked_fit_max_neighbors_each_side"),
        )
    )
    print(
        "[mapping_server]   training/input "
        + _format_cfg_items(
            use_norm_stats=_cfg_value(cfg, "use_norm_stats"),
            hz_randomization=_cfg_value(cfg, "hz_randomization_enable"),
            hz_choices=_cfg_value(cfg, "hz_randomization_choices"),
            hz_filter=_cfg_value(cfg, "hz_filter"),
            dq_delay_ms=_cfg_value(cfg, "dq_delay_range_ms"),
            torque_delay_ms=_cfg_value(cfg, "torque_delay_range_ms"),
            ideal_model_contract="public_tam_real_gravity",
        )
    )
    print(
        "[mapping_server]   runtime "
        + _format_cfg_items(
            expected_dt_s=args.expected_dt,
            patch_size=getattr(adaptor, "_patch_size", "n/a"),
            patch_stride=getattr(adaptor, "_patch_stride", "n/a"),
            decode_patch_size=getattr(adaptor, "_decode_patch_size", "n/a"),
            context_half=getattr(adaptor, "_context_half", "n/a"),
            attention_history_s=getattr(args, "attention_history_s", None),
            dagger_attention_history_s=_cfg_value(
                dagger_cfg,
                "attention_history_s",
            ),
            attention_history_tokens=getattr(adaptor, "_attention_history_tokens", "n/a"),
            attention_history_cache_tokens=getattr(
                adaptor,
                "_attention_history_cache_tokens",
                "n/a",
            ),
            require_explicit_fused_history=getattr(
                args,
                "require_explicit_fused_history",
                True,
            ),
            smoothing=getattr(adaptor, "_history_smoothing", "masked_local_fit"),
            jax_cache_dir=getattr(adaptor, "_jax_cache_dir", None) or "disabled",
            jax_cache_min_compile_time_s=args.history_jax_cache_min_compile_time_s,
            jax_cache_min_entry_size_bytes=args.history_jax_cache_min_entry_size_bytes,
            min_patches_before_send=args.min_patches_before_send,
            embedding_interval_s=args.embedding_interval_s,
            enable_after_first_embedding=args.enable_after_first_embedding,
            require_control_enable=args.require_control_enable,
            reset_on_controller_reset=args.reset_on_controller_reset,
        )
    )
    print(
        "[mapping_server]   controller "
        + _format_cfg_items(
            send_bin=args.send_bin,
            bin_name=bin_name or "n/a",
            bin_base64_bytes=len(bin_b64 or ""),
            xml_path=getattr(inf, "xml_path", "n/a"),
            command_endpoint=args.command_endpoint,
            request_endpoint=args.request_endpoint,
            history_endpoint=args.history_endpoint,
        )
    )


class _KeyboardShortcutMonitor:
    def __init__(self, stream: Any | None = None) -> None:
        self._stream = sys.stdin if stream is None else stream
        self._fd: Optional[int] = None
        self._old_termios: Any = None
        self.enabled = False

    def __enter__(self) -> "_KeyboardShortcutMonitor":
        if (
            self._stream is None
            or select is None
            or termios is None
            or tty is None
            or not hasattr(self._stream, "isatty")
            or not self._stream.isatty()
        ):
            return self
        try:
            self._fd = int(self._stream.fileno())
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except Exception:
            self._fd = None
            self._old_termios = None
            self.enabled = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def poll_keys(self) -> list[str]:
        if not self.enabled or self._stream is None or select is None:
            return []
        keys: list[str] = []
        try:
            while True:
                ready, _, _ = select.select([self._stream], [], [], 0.0)
                if not ready:
                    break
                char = self._stream.read(1)
                if not char:
                    break
                keys.append(str(char))
        except Exception:
            return keys
        return keys

    def close(self) -> None:
        if self.enabled and self._fd is not None and self._old_termios is not None and termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        self._fd = None
        self._old_termios = None
        self.enabled = False


def _latest_controller_timestamp(window: Sequence[dict]) -> Optional[float]:
    for sample in reversed(window):
        if not isinstance(sample, dict):
            continue
        for key in ("t", "t_raw", "timestamp"):
            value = sample.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
    return None


def _detect_controller_restart_reason(
    *,
    latest_controller_time: Optional[float],
    previous_controller_time: Optional[float],
    rewind_tolerance_s: float = 0.05,
) -> Optional[str]:
    if latest_controller_time is None or previous_controller_time is None:
        return None
    if float(latest_controller_time) + float(rewind_tolerance_s) < float(previous_controller_time):
        return (
            "controller_time_rewind:"
            f"{float(previous_controller_time):.6f}->{float(latest_controller_time):.6f}"
        )
    return None


def _window_to_arrays(window: Sequence[dict]) -> Optional[tuple[np.ndarray, ...]]:
    return extract_history_window_arrays(
        window,
        dof=7,
        q_keys=("q", "qpos"),
        dq_keys=("dq", "qd", "qvel"),
        tau_keys=("tau_cmd", "tau_commanded", "tau", "u_des", "u", "tau_measured"),
        gravity_keys=("gravity",),
        t_keys=("t", "t_raw", "timestamp"),
        valid_keys=("valid_for_history",),
    )


@dataclass(frozen=True)
class FusedHistoryWindowArrays:
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau_applied: np.ndarray
    tau_base: np.ndarray
    tau_tam: np.ndarray
    gravity: Optional[np.ndarray]
    keep: Optional[np.ndarray]
    source_labels: tuple[str, ...]
    missing_delta_rows: int


def _first_history_array(
    sample: dict,
    keys: Sequence[str],
    *,
    dof: int,
) -> Optional[np.ndarray]:
    for key in keys:
        if key not in sample or sample[key] is None:
            continue
        arr = np.asarray(sample[key], dtype=np.float32).reshape(-1)
        if arr.size >= int(dof):
            return arr[: int(dof)]
    return None


def _history_sample_timestamp(sample: dict) -> Optional[float]:
    for key in ("t", "t_raw", "timestamp"):
        if key not in sample or sample[key] is None:
            continue
        try:
            return float(sample[key])
        except Exception:
            continue
    return None


def _history_sample_keep(sample: dict) -> tuple[float, bool]:
    for key in ("valid_for_history",):
        if key in sample and sample[key] is not None:
            return (1.0 if bool(sample[key]) else 0.0), True
    return 1.0, False


def _derive_base_tam_torques(
    sample: dict,
    tau_applied: np.ndarray,
    *,
    dof: int,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    tau_base = _first_history_array(sample, _BASE_TAU_KEYS, dof=dof)
    tau_tam = _first_history_array(sample, _TAM_RESIDUAL_TAU_KEYS, dof=dof)
    applied = np.asarray(tau_applied, dtype=np.float32).reshape(int(dof))
    if tau_base is not None and tau_tam is not None:
        if sample.get("publish_ready") is True:
            return tau_base, tau_tam, "explicit_base_and_tam", False
        return tau_base, tau_tam, "explicit_base_and_tam_unverified", False
    if tau_tam is not None:
        return applied - tau_tam, tau_tam, "applied_minus_tam_delta", False
    if tau_base is not None:
        return tau_base, applied - tau_base, "applied_minus_explicit_base", False
    return applied.copy(), np.zeros_like(applied), "missing_delta_zero", True


def _window_to_fused_history_arrays(
    window: Sequence[dict],
    *,
    dof: int = 7,
) -> Optional[FusedHistoryWindowArrays]:
    ts_list: list[float] = []
    q_list: list[np.ndarray] = []
    dq_list: list[np.ndarray] = []
    tau_applied_list: list[np.ndarray] = []
    tau_base_list: list[np.ndarray] = []
    tau_tam_list: list[np.ndarray] = []
    gravity_list: list[np.ndarray] = []
    keep_list: list[float] = []
    source_labels: list[str] = []
    have_all_gravity = True
    have_any_keep = False
    missing_delta_rows = 0

    for sample in window:
        if not isinstance(sample, dict):
            continue
        t = _history_sample_timestamp(sample)
        if t is None:
            continue
        q = _first_history_array(sample, ("q", "qpos"), dof=dof)
        dq = _first_history_array(sample, ("dq", "qd", "qvel"), dof=dof)
        tau_applied = _first_history_array(sample, _APPLIED_TAU_KEYS, dof=dof)
        if q is None or dq is None or tau_applied is None:
            continue
        tau_base, tau_tam, source_label, missing_delta = _derive_base_tam_torques(
            sample,
            tau_applied,
            dof=dof,
        )
        gravity = _first_history_array(sample, ("gravity",), dof=dof)
        keep_value, has_keep = _history_sample_keep(sample)

        ts_list.append(float(t))
        q_list.append(q)
        dq_list.append(dq)
        tau_applied_list.append(tau_applied)
        tau_base_list.append(tau_base)
        tau_tam_list.append(tau_tam)
        keep_list.append(float(keep_value))
        source_labels.append(source_label)
        missing_delta_rows += int(missing_delta)
        have_any_keep = have_any_keep or has_keep
        if gravity is None:
            have_all_gravity = False
            gravity_list.append(np.zeros((int(dof),), dtype=np.float32))
        else:
            gravity_list.append(gravity)

    if not ts_list:
        return None
    return FusedHistoryWindowArrays(
        t=np.asarray(ts_list, dtype=np.float64),
        q=np.asarray(q_list, dtype=np.float32),
        dq=np.asarray(dq_list, dtype=np.float32),
        tau_applied=np.asarray(tau_applied_list, dtype=np.float32),
        tau_base=np.asarray(tau_base_list, dtype=np.float32),
        tau_tam=np.asarray(tau_tam_list, dtype=np.float32),
        gravity=np.asarray(gravity_list, dtype=np.float32) if have_all_gravity else None,
        keep=np.asarray(keep_list, dtype=np.float32) if have_any_keep else None,
        source_labels=tuple(source_labels),
        missing_delta_rows=int(missing_delta_rows),
    )


def _sync_controller_ideal_model_has_gravity(
    client: HistoryControllerClient,
    enabled: bool,
    *,
    log_prefix: str,
) -> None:
    ok = client.request_set_ideal_model_has_gravity(bool(enabled))
    if not ok:
        raise RuntimeError(
            f"Controller rejected ideal_model_has_gravity={bool(enabled)}"
        )
    print(
        f"{log_prefix} Synced controller ideal_model_has_gravity={bool(enabled)}."
    )


def _export_adaptor_bin_blob(
    *,
    adaptor: Any,
    bin_out: Path,
    tag: str,
) -> tuple[str, str]:
    bin_out = bin_out.expanduser().resolve()
    bin_out.mkdir(parents=True, exist_ok=True)
    bin_path = bin_out / f"{tag}_export.bin"
    adaptor.inf.export_simadaptor_weights_cpp(bin_path)
    return bin_path.name, base64.b64encode(bin_path.read_bytes()).decode("ascii")


def _describe_adaptor_bin_blob(bin_b64: str) -> str:
    try:
        raw = base64.b64decode(bin_b64.encode("ascii"), validate=True)
    except Exception:
        return "header=unavailable"
    if len(raw) < 16:
        return f"header=truncated bytes={len(raw)}"
    header_bytes = min((len(raw) // 4) * 4, 24)
    ints = np.frombuffer(raw[:header_bytes], dtype="<i4")
    dof = int(ints[0])
    emb = int(ints[1])
    hidden = int(ints[2])
    depth = int(ints[3])
    fields = [
        f"bytes={len(raw)}",
        f"dof={dof}",
        f"emb={emb}",
        f"hidden={hidden}",
        f"depth={depth}",
    ]
    if ints.size >= 5:
        fields.append(f"history={int(ints[4])}")
    if ints.size >= 6:
        flags = int(ints[5])
        flag_names = []
        if flags & 2:
            flag_names.append("jointwise")
        if flags & 8:
            flag_names.append("jointwise_direct")
        if flags & 16:
            flag_names.append("command_conditioned")
        fields.append(
            f"flags={flags}"
            + (f"({','.join(flag_names)})" if flag_names else "")
        )
    return " ".join(fields)


def _adaptor_bin_rejection_hint() -> str:
    return (
        "This usually means the controller-side history publisher build is "
        "stale for this TAM bin layout. Rebuild it from a version that "
        "supports command-conditioned/jointwise v2 adaptor bins, then "
        "restart the controller-side bridge."
    )


def _upload_adaptor_bin_blob(
    client: HistoryControllerClient,
    *,
    bin_name: str,
    bin_b64: str,
    log_prefix: str,
    enable_after_load: bool = False,
) -> None:
    print(
        f"{log_prefix} Uploading adaptor bin {bin_name} "
        f"({len(bin_b64)} base64 chars; {_describe_adaptor_bin_blob(bin_b64)})..."
    )
    response_fn = getattr(client, "request_load_bin_blob_response", None)
    if callable(response_fn):
        try:
            resp = response_fn(bin_name, bin_b64, enable_after_load=bool(enable_after_load))
        except TypeError:
            resp = response_fn(bin_name, bin_b64)
        ok = bool(resp.get("ok", False))
        error = resp.get("error")
    else:
        try:
            ok = client.request_load_bin_blob(
                bin_name,
                bin_b64,
                enable_after_load=bool(enable_after_load),
            )
        except TypeError:
            ok = client.request_load_bin_blob(bin_name, bin_b64)
        resp = {}
        error = None
    if not ok:
        detail = f"Controller rejected adaptor bin upload: {bin_name}"
        if error:
            detail += f" ({error})"
        raise AdaptorBinUploadError(f"{detail}. {_adaptor_bin_rejection_hint()}")
    if (not bool(enable_after_load)) and bool(resp.get("adaptor_enabled", False)):
        print(
            f"{log_prefix} Warning: controller reported adaptor_enabled=True after bin load; "
            "forcing adaptor disabled."
        )
    print(f"{log_prefix} Controller accepted adaptor bin upload: {bin_name}")


def _send_history_embedding(
    client: HistoryControllerClient,
    history_emb: np.ndarray,
    *,
    reliable: bool,
    log_prefix: str,
) -> bool:
    emb_flat = flatten_history_embedding_for_transport(
        np.asarray(history_emb, dtype=np.float32)
    )
    if reliable:
        try:
            ok = client.request_set_embedding(emb_flat)
        except Exception as exc:
            print(
                f"{log_prefix} Reliable embedding send failed; "
                f"falling back to best-effort: {exc}"
            )
        else:
            if ok:
                return True
            print(
                f"{log_prefix} Controller rejected reliable embedding send; "
                "falling back to best-effort."
            )
    client.set_embedding_best_effort(emb_flat)
    return False


def _clear_history_embedding(
    client: HistoryControllerClient,
    *,
    reliable: bool,
    log_prefix: str,
) -> bool:
    return _send_history_embedding(
        client,
        np.zeros((0,), dtype=np.float32),
        reliable=reliable,
        log_prefix=log_prefix,
    )


def _set_adaptor_enabled(
    client: HistoryControllerClient,
    enabled: bool,
    *,
    reliable: bool,
    log_prefix: str,
) -> bool:
    if reliable:
        try:
            ok = client.request_enable_adaptor(enabled)
        except Exception as exc:
            print(
                f"{log_prefix} Reliable enable_adaptor failed; "
                f"falling back to best-effort: {exc}"
            )
        else:
            if ok:
                return True
            print(
                f"{log_prefix} Controller rejected reliable enable_adaptor; "
                "falling back to best-effort."
            )
    client.enable_adaptor_best_effort(enabled)
    return False


@dataclass
class SimAdaptorMappingUpdater:
    client: HistoryControllerClient
    adaptor: Any
    embedding_interval_s: float
    min_patches_before_send: int
    enable_after_first_embedding: bool
    reset_on_controller_reset: bool
    print_embedding: bool
    ideal_model_has_gravity: bool
    history_torque_mode: str = HISTORY_TORQUE_MODE_APPLIED
    base_history_adaptor: Any | None = None
    tam_history_adaptor: Any | None = None
    history_fusion_params: Any | None = None
    require_explicit_fused_history: bool = True
    require_control_enable: bool = False
    bin_name: Optional[str] = None
    bin_b64: Optional[str] = None
    simadaptor_checkpoint: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False
    control_enable_allowed: bool = True
    num_windows: int = 0
    num_valid: int = 0
    num_emb: int = 0
    num_sent: int = 0
    patches_since_reset: int = 0
    last_window_wall: Optional[float] = None
    last_valid_wall: Optional[float] = None
    last_embedding_wall: Optional[float] = None
    last_send_wall: Optional[float] = None
    last_controller_time: Optional[float] = None
    pending_remote_prepare: bool = False
    last_prepare_attempt_wall: Optional[float] = None
    remote_prepare_failed: bool = False
    remote_prepare_error: Optional[str] = None
    pending_embedding: Optional[np.ndarray] = None
    current_embedding: Optional[np.ndarray] = None
    enable_hold_until_wall: Optional[float] = None
    bin_uploaded_once: bool = False
    fused_windows: int = 0
    fused_missing_delta_rows: int = 0
    fused_rejected_windows: int = 0
    fused_source_counts: dict[str, int] = field(default_factory=dict)
    fused_contract_valid: Optional[bool] = None
    warned_missing_fused_delta: bool = False
    warned_nonexplicit_fused_history: bool = False

    def __post_init__(self) -> None:
        self.embedding_interval_s = float(self.embedding_interval_s)
        self.min_patches_before_send = int(self.min_patches_before_send)
        if self.min_patches_before_send < 0:
            raise ValueError(
                "min_patches_before_send must be >= 0, "
                f"got {self.min_patches_before_send}"
            )
        self.enable_after_first_embedding = bool(self.enable_after_first_embedding)
        self.reset_on_controller_reset = bool(self.reset_on_controller_reset)
        self.print_embedding = bool(self.print_embedding)
        self.ideal_model_has_gravity = bool(self.ideal_model_has_gravity)
        self.require_explicit_fused_history = bool(self.require_explicit_fused_history)
        self.history_torque_mode = str(
            self.history_torque_mode or HISTORY_TORQUE_MODE_APPLIED
        ).strip()
        if self.history_torque_mode not in (
            HISTORY_TORQUE_MODE_APPLIED,
            HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
        ):
            raise ValueError(
                "history_torque_mode must be 'applied' or 'base_tam_fusion', "
                f"got {self.history_torque_mode!r}."
            )
        if self.history_torque_mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
            if self.base_history_adaptor is None or self.tam_history_adaptor is None:
                raise ValueError(
                    "base_tam_fusion requires base_history_adaptor and tam_history_adaptor."
                )
            if self.history_fusion_params is None:
                raise ValueError("base_tam_fusion requires history_fusion_params.")
        self.require_control_enable = bool(self.require_control_enable)
        self.control_enable_allowed = not self.require_control_enable
        self.enabled = self.control_enable_allowed and not self.delay_enable

    @property
    def delay_enable(self) -> bool:
        return bool(
            self.enable_after_first_embedding or self.min_patches_before_send > 0
        )

    def hold_enable_for(self, delay_s: float, *, now: Optional[float] = None) -> None:
        if now is None:
            now = time.perf_counter()
        delay_s = max(float(delay_s), 0.0)
        self.control_enable_allowed = True
        self.enable_hold_until_wall = float(now) + delay_s if delay_s > 0.0 else None
        if self.enable_hold_until_wall is not None:
            self.enabled = False
            used_reliable = _set_adaptor_enabled(
                self.client,
                False,
                reliable=True,
                log_prefix="[mapping_server]",
            )
            transport = "reliable" if used_reliable else "best-effort"
            print(
                "[mapping_server] Delaying TAM enable for "
                f"{delay_s:.3f}s while live updates continue "
                f"(disabled via {transport})."
            )

    def allow_control_enable(
        self,
        *,
        delay_s: float = 0.0,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.perf_counter()
        delay_s = max(float(delay_s), 0.0)
        self.control_enable_allowed = True
        if delay_s > 0.0:
            self.hold_enable_for(delay_s, now=now)

    def _enable_hold_remaining_s(self, *, now: Optional[float] = None) -> float:
        if self.enable_hold_until_wall is None:
            return 0.0
        if now is None:
            now = time.perf_counter()
        remaining = float(self.enable_hold_until_wall) - float(now)
        if remaining <= 0.0:
            self.enable_hold_until_wall = None
            return 0.0
        return remaining

    def _enable_hold_active(self, *, now: Optional[float] = None) -> bool:
        return self._enable_hold_remaining_s(now=now) > 0.0

    def _maybe_enable_adaptor(
        self,
        *,
        now: Optional[float] = None,
        reliable: bool,
        action_text: str,
    ) -> bool:
        if self.enabled:
            self._enable_hold_remaining_s(now=now)
            return False
        if not self.control_enable_allowed:
            return False
        if self._enable_hold_active(now=now):
            return False
        _set_adaptor_enabled(
            self.client,
            True,
            reliable=bool(reliable),
            log_prefix="[mapping_server]",
        )
        self.enabled = True
        self.enable_hold_until_wall = None
        print(f"[mapping_server] Adaptor enabled {action_text}.")
        return True

    def prepare_remote_controller(self) -> None:
        now = time.perf_counter()
        _sync_controller_ideal_model_has_gravity(
            self.client,
            self.ideal_model_has_gravity,
            log_prefix="[mapping_server]",
        )
        if self.bin_name is not None and self.bin_b64 is not None and not self.bin_uploaded_once:
            _set_adaptor_enabled(
                self.client,
                False,
                reliable=True,
                log_prefix="[mapping_server]",
            )
            _upload_adaptor_bin_blob(
                self.client,
                bin_name=self.bin_name,
                bin_b64=self.bin_b64,
                log_prefix="[mapping_server]",
                enable_after_load=False,
            )
            self.bin_uploaded_once = True
            _set_adaptor_enabled(
                self.client,
                False,
                reliable=True,
                log_prefix="[mapping_server]",
            )
        elif self.bin_name is not None and self.bin_b64 is not None:
            print(
                "[mapping_server] Reusing previously uploaded adaptor bin; "
                "skipping reload to avoid transient controller-side enable."
            )
        _clear_history_embedding(
            self.client,
            reliable=True,
            log_prefix="[mapping_server]",
        )
        hold_active = self._enable_hold_active(now=now)
        self.enabled = self.control_enable_allowed and (not self.delay_enable) and not hold_active
        if self.delay_enable or hold_active or not self.control_enable_allowed:
            _set_adaptor_enabled(
                self.client,
                False,
                reliable=bool(hold_active),
                log_prefix="[mapping_server]",
            )
        self.pending_remote_prepare = False
        self.last_prepare_attempt_wall = None
        self.remote_prepare_failed = False
        self.remote_prepare_error = None

    def _invalidate_fused_history_contract(self) -> None:
        """Disable stale output and require a fresh fused context after a bad row."""
        if self.fused_contract_valid is False:
            return
        self.fused_contract_valid = False
        self.adaptor.reset()
        if self.base_history_adaptor is not None:
            self.base_history_adaptor.reset()
        if self.tam_history_adaptor is not None:
            self.tam_history_adaptor.reset()
        self.last_embedding_wall = None
        self.last_send_wall = None
        self.patches_since_reset = 0
        self.num_emb = 0
        self.num_sent = 0
        self.pending_embedding = None
        self.current_embedding = None
        _set_adaptor_enabled(
            self.client,
            False,
            reliable=True,
            log_prefix="[mapping_server]",
        )
        self.enabled = False

    def _record_permanent_prepare_failure(self, exc: Exception) -> None:
        self.pending_remote_prepare = False
        self.last_prepare_attempt_wall = None
        self.remote_prepare_failed = True
        self.remote_prepare_error = str(exc)
        self.enabled = False
        try:
            self.client.enable_adaptor_best_effort(False)
        except Exception:
            pass

    def _maybe_prepare_remote_controller(self, *, now: float) -> bool:
        if self.remote_prepare_failed:
            return False
        if not self.pending_remote_prepare:
            return True
        if (
            self.last_prepare_attempt_wall is not None
            and (float(now) - float(self.last_prepare_attempt_wall)) < _REMOTE_PREPARE_RETRY_INTERVAL_S
        ):
            return False
        self.last_prepare_attempt_wall = float(now)
        try:
            self.prepare_remote_controller()
        except AdaptorBinUploadError as exc:
            self._record_permanent_prepare_failure(exc)
            print(
                "[mapping_server] Controller rejected adaptor prepare; "
                "not retrying this bin until a controller reset/restart: "
                f"{exc}"
            )
            return False
        except Exception as exc:
            self.pending_remote_prepare = True
            print(
                "[mapping_server] Controller command interface not ready yet; "
                "waiting for more history before retrying adaptor prepare: "
                f"{exc}"
            )
            return False
        print("[mapping_server] Controller command interface recovered; adaptor prepare complete.")
        return True

    def handle_reset_event(
        self,
        event: dict[str, Any],
        *,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.perf_counter()
        print(f"[mapping_server] Controller reset: {event}")
        if not self.reset_on_controller_reset:
            return
        self.adaptor.reset()
        if self.base_history_adaptor is not None:
            self.base_history_adaptor.reset()
        if self.tam_history_adaptor is not None:
            self.tam_history_adaptor.reset()
        self.last_window_wall = None
        self.last_valid_wall = None
        self.last_embedding_wall = None
        self.last_send_wall = None
        self.last_controller_time = None
        self.patches_since_reset = 0
        self.num_emb = 0
        self.num_sent = 0
        if self.require_control_enable:
            self.control_enable_allowed = bool(
                event.get("allow_enable", event.get("direct", False))
            )
        self.enabled = (
            self.control_enable_allowed
            and (not self.delay_enable)
            and not self._enable_hold_active(now=now)
        )
        self.pending_embedding = None
        self.current_embedding = None
        self.pending_remote_prepare = True
        if not bool(event.get("direct", False)):
            self.bin_uploaded_once = False
        self.last_prepare_attempt_wall = None
        self.remote_prepare_failed = False
        self.remote_prepare_error = None
        self.fused_contract_valid = None

    def _push_applied_history_window(
        self,
        *,
        timestamps: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        gravity: Optional[np.ndarray],
        keep_mask: Optional[np.ndarray],
    ) -> Any | None:
        return self.adaptor.push_window(
            timestamps,
            q,
            dq,
            tau,
            gravity=gravity,
            keep_mask=keep_mask,
        )

    def _push_fused_history_window(
        self,
        fused: FusedHistoryWindowArrays,
        *,
        now: float,
    ) -> Any | None:
        del now
        self.fused_windows += 1
        self.fused_missing_delta_rows += int(fused.missing_delta_rows)
        for label in fused.source_labels:
            self.fused_source_counts[str(label)] = (
                int(self.fused_source_counts.get(str(label), 0)) + 1
            )
        nonexplicit_rows = sum(
            1 for label in fused.source_labels if label != "explicit_base_and_tam"
        )
        if self.require_explicit_fused_history and nonexplicit_rows > 0:
            self.fused_rejected_windows += 1
            self._invalidate_fused_history_contract()
            if not self.warned_nonexplicit_fused_history:
                print(
                    "[mapping_server] Refusing fused history until the controller-side "
                    "history publisher provides publish-ready rows with explicit "
                    "tau_base and tau_adaptor_delta fields: "
                    f"nonexplicit_rows={nonexplicit_rows}/{len(fused.source_labels)} "
                    f"sources={dict(self.fused_source_counts)}."
                )
                self.warned_nonexplicit_fused_history = True
            return None
        self.fused_contract_valid = True
        if fused.missing_delta_rows > 0 and not self.warned_missing_fused_delta:
            print(
                "[mapping_server] base_tam_fusion history: "
                f"{int(fused.missing_delta_rows)} row(s) in the first fused window "
                "had no TAM residual field; using zero residual for those rows."
            )
            self.warned_missing_fused_delta = True
        applied_emb = self.adaptor.push_window(
            fused.t,
            fused.q,
            fused.dq,
            fused.tau_applied,
            gravity=fused.gravity,
            raw_tau=fused.tau_applied,
            keep_mask=fused.keep,
        )
        base_emb = self.base_history_adaptor.push_window(
            fused.t,
            fused.q,
            fused.dq,
            fused.tau_base,
            gravity=fused.gravity,
            raw_tau=fused.tau_applied,
            keep_mask=fused.keep,
        )
        tam_emb = self.tam_history_adaptor.push_window(
            fused.t,
            fused.q,
            fused.dq,
            fused.tau_tam,
            tau_is_model_space=True,
            raw_tau=fused.tau_applied,
            keep_mask=fused.keep,
        )
        if applied_emb is None or base_emb is None or tam_emb is None:
            return None
        return _apply_history_fusion(
            self.history_fusion_params,
            applied_emb,
            base_emb,
            tam_emb,
        )

    def handle_window(
        self,
        window: Sequence[dict],
        *,
        now: Optional[float] = None,
    ) -> bool:
        if now is None:
            now = time.perf_counter()
        if not window:
            return False

        latest_controller_time = _latest_controller_timestamp(window)
        restart_reason = _detect_controller_restart_reason(
            latest_controller_time=latest_controller_time,
            previous_controller_time=self.last_controller_time,
        )
        if restart_reason is not None:
            self.handle_reset_event(
                {
                    "reason": restart_reason,
                    "synthetic": True,
                },
                now=now,
            )

        if self.num_windows == 0:
            print("[mapping_server] First history window received from controller.")
        self.num_windows += 1
        self.last_window_wall = now

        fused_arrays: Optional[FusedHistoryWindowArrays] = None
        if self.history_torque_mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
            fused_arrays = _window_to_fused_history_arrays(window)
            if fused_arrays is None:
                self.last_controller_time = latest_controller_time
                return False
            t_arr = fused_arrays.t
            q_arr = fused_arrays.q
            dq_arr = fused_arrays.dq
            tau_arr = fused_arrays.tau_applied
            gravity_arr = fused_arrays.gravity
            keep_arr = fused_arrays.keep
        else:
            arrays = _window_to_arrays(window)
            if arrays is None:
                self.last_controller_time = latest_controller_time
                return False

            t_arr, q_arr, dq_arr, tau_arr, gravity_arr, keep_arr = arrays
        if self.num_valid == 0:
            print(
                "[mapping_server] First valid history samples received: "
                f"{t_arr.shape[0]} rows with q/dq/tau "
                f"(history_torque_mode={self.history_torque_mode})."
            )
        self.num_valid += int(t_arr.shape[0])
        self.last_valid_wall = now
        self._maybe_prepare_remote_controller(now=now)

        t_used = (float(now) + (t_arr - t_arr[-1])).astype(np.float64)
        if fused_arrays is not None:
            fused_arrays = FusedHistoryWindowArrays(
                t=t_used,
                q=fused_arrays.q,
                dq=fused_arrays.dq,
                tau_applied=fused_arrays.tau_applied,
                tau_base=fused_arrays.tau_base,
                tau_tam=fused_arrays.tau_tam,
                gravity=fused_arrays.gravity,
                keep=fused_arrays.keep,
                source_labels=fused_arrays.source_labels,
                missing_delta_rows=fused_arrays.missing_delta_rows,
            )
            emb = self._push_fused_history_window(fused_arrays, now=now)
        else:
            emb = self._push_applied_history_window(
                timestamps=t_used,
                q=q_arr,
                dq=dq_arr,
                tau=tau_arr,
                gravity=gravity_arr,
                keep_mask=keep_arr,
            )
        if emb is not None:
            emb_np = np.asarray(emb, dtype=np.float32)
            if self.num_emb == 0:
                emb_dim = int(emb_np.reshape(-1).size)
                print(f"[mapping_server] First embedding ready (dim={emb_dim}).")
            self.num_emb += 1
            self.patches_since_reset += 1
            self.last_embedding_wall = now
            self.pending_embedding = emb_np
            self.current_embedding = np.array(emb_np, dtype=np.float32, copy=True)

        if self.patches_since_reset < self.min_patches_before_send:
            self.last_controller_time = latest_controller_time
            return False
        if self.last_send_wall is not None and (
            now - self.last_send_wall
        ) < self.embedding_interval_s:
            self.last_controller_time = latest_controller_time
            return False
        if self.pending_remote_prepare:
            self.last_controller_time = latest_controller_time
            return False
        if self.remote_prepare_failed:
            self.last_controller_time = latest_controller_time
            return False
        if self.pending_embedding is None:
            if (
                self.current_embedding is not None
                and self.num_sent > 0
                and not self.pending_remote_prepare
                and not self.remote_prepare_failed
            ):
                self._maybe_enable_adaptor(
                    now=now,
                    reliable=False,
                    action_text="after delayed enable hold",
                )
            self.last_controller_time = latest_controller_time
            return False

        was_first_send = self.num_sent == 0
        used_reliable = _send_history_embedding(
            self.client,
            self.pending_embedding,
            reliable=was_first_send,
            log_prefix="[mapping_server]",
        )
        self.num_sent += 1
        if self.print_embedding:
            print(f"[mapping_server] Sent embedding (dim={self.pending_embedding.size}).")
        if not self.enabled:
            self._maybe_enable_adaptor(
                now=now,
                reliable=was_first_send,
                action_text="after first embedding send",
            )
        if was_first_send:
            transport = "reliable" if used_reliable else "best-effort"
            print(
                "[mapping_server] First embedding sent to controller "
                f"(dim={self.pending_embedding.size}, transport={transport})."
            )
        self.last_send_wall = now
        self.pending_embedding = None
        self.last_controller_time = latest_controller_time
        return True

    def resend_current_output(self, *, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.perf_counter()
        if self.current_embedding is None:
            print("[mapping_server] Hold resend skipped: no embedding has been identified yet.")
            return False
        if self.remote_prepare_failed:
            print(
                "[mapping_server] Hold resend skipped: adaptor prepare failed: "
                f"{self.remote_prepare_error}"
            )
            return False
        if self.pending_remote_prepare and not self._maybe_prepare_remote_controller(now=now):
            print("[mapping_server] Hold resend skipped: controller command interface still not ready.")
            return False
        was_first_send = self.num_sent == 0
        used_reliable = _send_history_embedding(
            self.client,
            self.current_embedding,
            reliable=False,
            log_prefix="[mapping_server]",
        )
        self.num_sent += 1
        if not self.enabled:
            self._maybe_enable_adaptor(
                now=now,
                reliable=False,
                action_text="during hold resend",
            )
        self.last_send_wall = now
        transport = "reliable" if used_reliable else "best-effort"
        print(
            "[mapping_server] Hold resend embedding "
            f"({transport}): {_describe_embedding_for_log(self.current_embedding)}"
        )
        if was_first_send:
            print("[mapping_server] First embedding reached controller during hold mode.")
        return True

    def disable_current_output(self, *, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.perf_counter()
        used_reliable = _set_adaptor_enabled(
            self.client,
            False,
            reliable=False,
            log_prefix="[mapping_server]",
        )
        self.enabled = False
        if self.require_control_enable:
            self.control_enable_allowed = False
        self.enable_hold_until_wall = None
        self.last_send_wall = now
        transport = "reliable" if used_reliable else "best-effort"
        print(
            "[mapping_server] TAM disabled from keyboard "
            f"({transport})."
        )
        return True

    def process_once(
        self,
        *,
        window: Optional[Sequence[dict]] = None,
        reset_event: Optional[dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> bool:
        if now is None:
            now = time.perf_counter()
        if reset_event is not None:
            self.handle_reset_event(reset_event, now=now)
        if window is None:
            return False
        return self.handle_window(window, now=now)

    def status_payload(self, *, now: Optional[float] = None) -> dict[str, Any]:
        if now is None:
            now = time.perf_counter()
        if self.remote_prepare_failed:
            health = "adaptor_prepare_failed"
        elif self.num_windows == 0:
            health = "waiting_for_history"
        elif self.pending_remote_prepare:
            health = "waiting_for_controller_ready"
        elif self.last_window_wall is not None and (now - self.last_window_wall) > 5.0:
            health = "stalled_no_recent_history"
        elif self.num_valid == 0:
            health = "history_missing_keys"
        elif (
            self.history_torque_mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION
            and self.require_explicit_fused_history
            and self.fused_contract_valid is False
        ):
            health = "fused_history_contract_missing"
        elif self.last_embedding_wall is None:
            health = "collecting_context"
        elif self.patches_since_reset < self.min_patches_before_send:
            health = "warming_up"
        elif not self.control_enable_allowed:
            health = "waiting_for_control_enable"
        elif self._enable_hold_active(now=now):
            health = "enable_delayed"
        elif self.num_sent == 0:
            health = "ready_to_send"
        else:
            health = "healthy"
        mapping_mode = MAPPING_MODE_SIMADAPTOR if self.enabled else MAPPING_MODE_NONE
        return {
            "mapping_mode": mapping_mode,
            "configured_mapping_mode": MAPPING_MODE_SIMADAPTOR,
            "backend": "tam",
            "tam_checkpoint": dict(self.simadaptor_checkpoint),
            "history_torque_mode": str(self.history_torque_mode),
            "fused_windows": int(self.fused_windows),
            "fused_missing_delta_rows": int(self.fused_missing_delta_rows),
            "fused_rejected_windows": int(self.fused_rejected_windows),
            "fused_contract_valid": self.fused_contract_valid,
            "fused_source_counts": dict(self.fused_source_counts),
            "require_explicit_fused_history": bool(self.require_explicit_fused_history),
            "health": health,
            "windows": int(self.num_windows),
            "valid_samples": int(self.num_valid),
            "embeddings": int(self.num_emb),
            "sent": int(self.num_sent),
            "enabled": bool(self.enabled),
            "require_control_enable": bool(self.require_control_enable),
            "control_enable_allowed": bool(self.control_enable_allowed),
            "patches_since_reset": int(self.patches_since_reset),
            "min_patches_before_send": int(self.min_patches_before_send),
            "delay_enable": bool(self.delay_enable),
            "enable_hold_active": bool(self._enable_hold_active(now=now)),
            "enable_hold_remaining_s": float(self._enable_hold_remaining_s(now=now)),
            "enable_hold_until_perf_s": (
                None
                if self.enable_hold_until_wall is None
                else float(self.enable_hold_until_wall)
            ),
            "pending_remote_prepare": bool(self.pending_remote_prepare),
            "remote_prepare_failed": bool(self.remote_prepare_failed),
            "remote_prepare_error": self.remote_prepare_error,
            "last_window_age_s": _age_s(now, self.last_window_wall),
            "last_valid_age_s": _age_s(now, self.last_valid_wall),
            "last_embedding_age_s": _age_s(now, self.last_embedding_wall),
            "last_send_age_s": _age_s(now, self.last_send_wall),
            "last_window_age": _status_age(now, self.last_window_wall),
            "last_valid_age": _status_age(now, self.last_valid_wall),
            "last_embedding_age": _status_age(now, self.last_embedding_wall),
            "last_send_age": _status_age(now, self.last_send_wall),
        }

    def status_line(self, *, now: Optional[float] = None) -> str:
        if now is None:
            now = time.perf_counter()
        status = self.status_payload(now=now)
        return (
            "[mapping_server] "
            f"backend={status['backend']} "
            f"status={status['health']} "
            f"windows={status['windows']} "
            f"valid_samples={status['valid_samples']} "
            f"embeddings={status['embeddings']} "
            f"sent={status['sent']} "
            f"enabled={'yes' if status['enabled'] else 'no'} "
            f"history_torque_mode={status['history_torque_mode']} "
            f"patches={status['patches_since_reset']}/{status['min_patches_before_send']} "
            f"last_window={status['last_window_age']} "
            f"last_valid={status['last_valid_age']} "
            f"last_embedding={status['last_embedding_age']} "
            f"last_send={status['last_send_age']}"
        )

    def close(self) -> None:
        return None



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Workstation-side mapping daemon that consumes NUC history windows "
            "and sends TAM embeddings or exported TAM weights back to the "
            "controller bridge."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tam",
        choices=["tam"],
        help="Mapping backend to run.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help=(
            "Local TAM checkpoint path: either a run directory containing "
            "save_dict.pkl or a specific checkpoint_<step> directory."
        ),
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("assets/franka_panda/panda_pandagripper.xml"),
        help="MuJoCo XML used to resolve the robot model.",
    )
    parser.add_argument(
        "--history-endpoint",
        type=str,
        default=HISTORY_ENDPOINT,
        help=f"Controller PUB endpoint (default: {HISTORY_ENDPOINT}).",
    )
    parser.add_argument(
        "--command-endpoint",
        type=str,
        default=COMMAND_ENDPOINT,
        help=f"Controller PUSH/PULL command endpoint (default: {COMMAND_ENDPOINT}).",
    )
    parser.add_argument(
        "--request-endpoint",
        type=str,
        default=REQUEST_ENDPOINT,
        help=f"Controller REQ/REP endpoint (default: {REQUEST_ENDPOINT}).",
    )
    parser.add_argument(
        "--control-endpoint",
        type=str,
        default=DEFAULT_MAPPING_CONTROL_ENDPOINT,
        help=(
            "REQ/REP endpoint for direct mapping_server commands such as reset. "
            "Set empty to disable."
        ),
    )
    parser.add_argument(
        "--history-buffer",
        type=int,
        default=500,
        help="Rolling history buffer size for the client.",
    )
    parser.add_argument(
        "--poll-timeout-ms",
        type=int,
        default=20,
        help="Poll timeout for history windows.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=5.0,
        help="Heartbeat interval for status prints; set <=0 to disable.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print extra waiting/status information.",
    )
    parser.add_argument(
        "--enable-viser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable live Viser visualization of joint state and torque overlays.",
    )
    parser.add_argument("--viser-host", type=str, default="0.0.0.0")
    parser.add_argument("--viser-port", type=int, default=4242)
    parser.add_argument("--viser-fps", type=float, default=30.0)
    parser.add_argument(
        "--viser-prefix",
        type=str,
        default="/panda",
        help="Root scene prefix for the robot in Viser.",
    )
    parser.add_argument(
        "--viser-tcp-link",
        type=str,
        default="fts300_sensor_body",
        help="URDF link name used as TCP for force/torque arrows.",
    )
    parser.add_argument(
        "--viser-force-scale",
        type=float,
        default=2.0,
        help="Scale factor for Franka force arrows.",
    )
    parser.add_argument(
        "--viser-torque-scale",
        type=float,
        default=2.0,
        help="Scale factor for Franka torque arrows.",
    )
    parser.add_argument(
        "--viser-arrow-radius",
        type=float,
        default=0.08,
        help="Arrow radius for force/torque visualization.",
    )
    parser.add_argument(
        "--viser-invert-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert displayed force/torque vectors in Viser.",
    )

    parser.add_argument(
        "--expected-dt",
        type=float,
        default=0.001,
        help="Fallback dt (s) for the runtime adaptor.",
    )
    parser.add_argument(
        "--attention-history-s",
        type=float,
        default=DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
        help=(
            "Limit autoregressive history attention to this many seconds. "
            f"Default: {DEFAULT_DEPLOY_ATTENTION_HISTORY_S:g}; <=0 keeps the full cache."
        ),
    )
    parser.add_argument(
        "--history-torque-mode",
        type=str,
        default=HISTORY_TORQUE_MODE_AUTO,
        choices=HISTORY_TORQUE_MODE_CHOICES,
        help=(
            "History torque contract for TAM checkpoints. 'auto' first reads "
            "online-DAgger metadata, then base config metadata, and finally infers "
            "base_tam_fusion from checkpoint history_fusion weights."
        ),
    )
    parser.add_argument(
        "--require-explicit-fused-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For base_tam_fusion checkpoints, require every controller history row "
            "to publish explicit tau_base and tau_adaptor_delta fields before sending "
            "or enabling an embedding."
        ),
    )
    parser.add_argument(
        "--history-jax-cache-dir",
        type=Path,
        default=Path(".cache") / "jax_history_compile",
        help=(
            "Persistent JAX compilation cache directory for the TAM "
            "history encoder. Reuse the same directory across the prewarm helper and "
            "mapping_server runs to avoid repeated decode-step compile latency."
        ),
    )
    parser.add_argument(
        "--history-jax-cache-min-compile-time-s",
        type=float,
        default=0.0,
        help=(
            "Minimum compile time threshold for persistent JAX cache entries. "
            "Default 0 caches the deploy history decoder even if JAX thinks it is small."
        ),
    )
    parser.add_argument(
        "--history-jax-cache-min-entry-size-bytes",
        type=int,
        default=-1,
        help=(
            "Minimum persistent-cache entry size in bytes. Default -1 asks JAX to "
            "cache all history-decoder executables that pass the compile-time threshold."
        ),
    )
    parser.add_argument(
        "--history-jax-cache-xla-caches",
        type=str,
        default="xla_gpu_per_fusion_autotune_cache_dir",
        help=(
            "Extra XLA caches to persist for history encoder compile/autotune reuse. "
            "Set empty to leave JAX's XLA-cache extension unset."
        ),
    )
    parser.add_argument(
        "--min-patches-before-send",
        type=int,
        default=2,
        help="Wait for this many decoded patches before sending/enabling the adaptor.",
    )
    parser.add_argument(
        "--embedding-interval-s",
        type=float,
        default=0.2,
        help="Minimum wall-time between embedding sends.",
    )
    parser.add_argument(
        "--enable-after-first-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptor only after sending the first computed embedding.",
    )
    parser.add_argument(
        "--require-control-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the TAM controller disabled until the mapping-server control "
            "endpoint receives reset/resume/hold. This prevents an already-running "
            "mapping_server from enabling TAM immediately when "
            "the history controller restarts before the robot has been homed."
        ),
    )
    parser.add_argument(
        "--reset-on-controller-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the runtime cache when the controller publishes a reset event.",
    )
    parser.add_argument(
        "--send-bin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export adaptor weights to .bin and upload them to the controller.",
    )
    parser.add_argument(
        "--bin-out",
        type=Path,
        default=Path("eval_logs") / "tmp_bin_exports",
        help="Directory to write the exported adaptor .bin before upload.",
    )
    parser.add_argument(
        "--print-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print a log line on every embedding send.",
    )

    return parser


def _resolve_simadaptor_source(args: argparse.Namespace) -> argparse.Namespace:
    if args.ckpt_path is None:
        raise SystemExit("Provide --ckpt-path.")
    return args


def _build_updater(
    args: argparse.Namespace,
    client: HistoryControllerClient,
) -> SimAdaptorMappingUpdater:
    args.backend = _normalize_backend_name(args.backend)
    if args.backend == "tam":
        args = _resolve_simadaptor_source(args)
        from simadaptor.deploy.history_runtime import RealTimeHistoryAdaptor

        adaptor = RealTimeHistoryAdaptor(
            simadaptor_ckpt_path=str(args.ckpt_path) if args.ckpt_path is not None else None,
            xml_path=args.xml,
            expected_dt=float(args.expected_dt),
            attention_history_s=args.attention_history_s,
            jax_cache_dir=args.history_jax_cache_dir,
            jax_cache_min_compile_time_s=float(args.history_jax_cache_min_compile_time_s),
            jax_cache_min_entry_size_bytes=int(args.history_jax_cache_min_entry_size_bytes),
            jax_cache_xla_caches=(
                str(args.history_jax_cache_xla_caches)
                if args.history_jax_cache_xla_caches
                else None
            ),
        )
        resolved_history_torque_mode = _resolve_simadaptor_history_torque_mode(
            adaptor,
            str(args.history_torque_mode),
        )
        args.resolved_history_torque_mode = resolved_history_torque_mode
        ablation_mode = _simadaptor_ablation_mode(adaptor)
        if ablation_mode != "tam":
            raise RuntimeError(
                "The public TAM mapping server supports only TAM checkpoints; "
                f"loaded cfg.ablation_mode={ablation_mode!r}."
            )

        bin_name = None
        bin_b64 = None
        if args.send_bin:
            tag = Path(args.ckpt_path).name if args.ckpt_path is not None else "tam"
            bin_name, bin_b64 = _export_adaptor_bin_blob(
                adaptor=adaptor,
                bin_out=args.bin_out,
                tag=tag,
            )
        _print_simadaptor_deploy_config(
            args=args,
            adaptor=adaptor,
            bin_name=bin_name,
            bin_b64=bin_b64,
        )
        simadaptor_checkpoint = _simadaptor_checkpoint_meta(
            args=args,
            adaptor=adaptor,
            bin_name=bin_name,
            bin_b64=bin_b64,
        )
        base_history_adaptor = None
        tam_history_adaptor = None
        history_fusion_params = None
        if resolved_history_torque_mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
            history_fusion_params = _simadaptor_history_fusion_params(adaptor)
            base_history_adaptor = RealTimeHistoryAdaptor(
                sim_inf=adaptor.inf,
                runtime_bundle=adaptor.runtime_bundle,
                expected_dt=float(args.expected_dt),
                attention_history_s=args.attention_history_s,
                jax_cache_dir=args.history_jax_cache_dir,
                jax_cache_min_compile_time_s=float(args.history_jax_cache_min_compile_time_s),
                jax_cache_min_entry_size_bytes=int(args.history_jax_cache_min_entry_size_bytes),
                jax_cache_xla_caches=(
                    str(args.history_jax_cache_xla_caches)
                    if args.history_jax_cache_xla_caches
                    else None
                ),
            )
            tam_history_adaptor = RealTimeHistoryAdaptor(
                sim_inf=adaptor.inf,
                runtime_bundle=adaptor.runtime_bundle,
                expected_dt=float(args.expected_dt),
                attention_history_s=args.attention_history_s,
                jax_cache_dir=args.history_jax_cache_dir,
                jax_cache_min_compile_time_s=float(args.history_jax_cache_min_compile_time_s),
                jax_cache_min_entry_size_bytes=int(args.history_jax_cache_min_entry_size_bytes),
                jax_cache_xla_caches=(
                    str(args.history_jax_cache_xla_caches)
                    if args.history_jax_cache_xla_caches
                    else None
                ),
            )
        return SimAdaptorMappingUpdater(
            client=client,
            adaptor=adaptor,
            embedding_interval_s=float(args.embedding_interval_s),
            min_patches_before_send=int(args.min_patches_before_send),
            enable_after_first_embedding=bool(args.enable_after_first_embedding),
            reset_on_controller_reset=bool(args.reset_on_controller_reset),
            print_embedding=bool(args.print_embedding),
            ideal_model_has_gravity=bool(adaptor.inf.ideal_model_has_gravity),
            history_torque_mode=resolved_history_torque_mode,
            base_history_adaptor=base_history_adaptor,
            tam_history_adaptor=tam_history_adaptor,
            history_fusion_params=history_fusion_params,
            require_explicit_fused_history=bool(args.require_explicit_fused_history),
            require_control_enable=bool(args.require_control_enable),
            bin_name=bin_name,
            bin_b64=bin_b64,
            simadaptor_checkpoint=simadaptor_checkpoint,
        )

    raise RuntimeError(f"Unsupported public TAM mapping backend: {args.backend!r}.")


def _bind_control_socket(endpoint: str) -> Optional[zmq.Socket]:
    endpoint = str(endpoint).strip()
    if not endpoint:
        return None
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.LINGER = 0
    sock.bind(endpoint)
    print(f"[mapping_server] Control endpoint listening on {endpoint}.")
    return sock


def _mapping_server_status_payload(
    *,
    updater: SimAdaptorMappingUpdater,
    args: argparse.Namespace,
    paused: bool,
    manually_disabled: bool,
    now: float,
) -> dict[str, Any]:
    status_fn = getattr(updater, "status_payload", None)
    if callable(status_fn):
        payload = dict(status_fn(now=now))
    else:
        payload = {}
    if not payload.get("mapping_mode"):
        payload["mapping_mode"] = _mapping_mode_for_backend(args.backend)
    payload.setdefault("backend", str(args.backend))
    payload.update(
        {
            "ok": True,
            "server_running": True,
            "control_endpoint": str(args.control_endpoint),
            "paused": bool(paused),
            "manually_disabled": bool(manually_disabled),
            "pause_state": "disabled" if manually_disabled else ("hold" if paused else "running"),
            "time_unix_s": float(time.time()),
            "perf_counter_s": float(now),
        }
    )
    return payload


def _poll_control_command(
    sock: Optional[zmq.Socket],
    *,
    status_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if sock is None:
        return None
    command_event: Optional[dict[str, Any]] = None
    while True:
        try:
            msg = sock.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        except Exception as exc:
            print(f"[mapping_server] Failed to read control command: {exc}")
            break
        if not isinstance(msg, dict):
            print(f"[mapping_server] Ignoring non-dict control command: {type(msg).__name__}")
            sock.send_json({"ok": False, "error": "control command must be a dict"})
            continue
        cmd = str(msg.get("cmd", "")).strip().lower()
        if cmd not in {"reset", "disable", "resume", "hold", "status"}:
            print(f"[mapping_server] Ignoring unknown control command: {cmd or '<empty>'}")
            sock.send_json({"ok": False, "error": f"unknown command {cmd or '<empty>'}"})
            continue
        if cmd == "status":
            sock.send_json(
                {
                    "ok": True,
                    "cmd": cmd,
                    "mapping_mode": (
                        str(status_payload.get("mapping_mode", "none"))
                        if isinstance(status_payload, dict)
                        else "none"
                    ),
                    "backend": (
                        status_payload.get("backend")
                        if isinstance(status_payload, dict)
                        else None
                    ),
                    "mapping_server": dict(status_payload or {}),
                }
            )
            continue
        enable_delay_s = None
        if "enable_delay_s" in msg:
            try:
                enable_delay_s = max(float(msg.get("enable_delay_s", 0.0) or 0.0), 0.0)
            except Exception:
                sock.send_json({"ok": False, "error": "enable_delay_s must be a nonnegative float"})
                continue
        allow_enable = msg.get("allow_enable", None)
        if allow_enable is not None:
            if isinstance(allow_enable, str):
                allow_enable = allow_enable.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                allow_enable = bool(allow_enable)
        command_event = {
            "cmd": cmd,
            "reason": msg.get("reason", f"mapping_server_control:{cmd}"),
            "timestamp": msg.get("timestamp", time.time()),
            "source": msg.get("source", "mapping_server_control"),
            "direct": True,
        }
        if enable_delay_s is not None:
            command_event["enable_delay_s"] = float(enable_delay_s)
        if allow_enable is not None:
            command_event["allow_enable"] = bool(allow_enable)
        sock.send_json(
            {
                "ok": True,
                "cmd": cmd,
                "reason": command_event["reason"],
                "enable_delay_s": enable_delay_s,
                "allow_enable": allow_enable,
            }
        )
    return command_event


def _apply_control_enable_delay(
    updater: SimAdaptorMappingUpdater,
    control_event: dict[str, Any],
    *,
    now: float,
) -> None:
    try:
        delay_s = float(control_event.get("enable_delay_s", 0.0) or 0.0)
    except Exception:
        delay_s = 0.0
    if control_event.get("allow_enable", True) is False:
        return
    allow_fn = getattr(updater, "allow_control_enable", None)
    if callable(allow_fn):
        allow_fn(delay_s=delay_s, now=now)
        return
    hold_fn = getattr(updater, "hold_enable_for", None)
    if delay_s > 0.0 and callable(hold_fn):
        hold_fn(delay_s, now=now)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.attention_history_s is not None and args.attention_history_s <= 0:
        # Documented escape hatch: values <= 0 keep the full decode cache.
        args.attention_history_s = None

    client = HistoryControllerClient(
        history_endpoint=args.history_endpoint,
        command_endpoint=args.command_endpoint,
        history_buffer=int(args.history_buffer),
        request_endpoint=args.request_endpoint,
    )
    if bool(args.enable_viser):
        client.enable_viewer(
            str(args.xml),
            host=str(args.viser_host),
            port=int(args.viser_port),
            prefix=str(args.viser_prefix),
            fps=float(args.viser_fps),
            tcp_link=str(args.viser_tcp_link),
            force_scale=float(args.viser_force_scale),
            torque_scale=float(args.viser_torque_scale),
            arrow_radius=float(args.viser_arrow_radius),
            invert_vectors=bool(args.viser_invert_vectors),
        )
        print(
            f"[mapping_server] Viser enabled at http://{args.viser_host}:{int(args.viser_port)} "
            f"(prefix={args.viser_prefix})."
        )
    updater = _build_updater(args, client)
    control_sock = _bind_control_socket(str(args.control_endpoint))

    try:
        try:
            updater.prepare_remote_controller()
        except AdaptorBinUploadError as exc:
            updater._record_permanent_prepare_failure(exc)
            print(
                "[mapping_server] Controller rejected adaptor prepare at startup; "
                "not retrying this bin until a controller reset/restart: "
                f"{exc}"
            )
        except Exception as exc:
            updater.pending_remote_prepare = True
            updater.last_prepare_attempt_wall = None
            print(
                "[mapping_server] Controller command interface not ready at startup; "
                "waiting for history before retrying prepare: "
                f"{exc}"
            )

        print(
            "[mapping_server] Listening for history; Ctrl-C to stop. "
            f"Backend={args.backend} "
            f"mapping_mode={_mapping_mode_for_backend(args.backend)}. "
            f"Heartbeat {_heartbeat_text(float(args.status_interval_s))}."
        )
        paused = False
        manually_disabled = False
        last_status_wall = time.perf_counter()
        last_hold_resend_wall: Optional[float] = None

        with _KeyboardShortcutMonitor() as key_monitor:
            if key_monitor.enabled:
                print(
                    "[mapping_server] Keyboard shortcuts: press 's' to hold live updates "
                    "and keep re-sending the current output; press 'd' to disable the active "
                    "backend; press 'a' to resume."
                )
            while True:
                for key in key_monitor.poll_keys():
                    key_lower = key.lower()
                    now = time.perf_counter()
                    if key_lower == "s":
                        manually_disabled = False
                        if not paused:
                            paused = True
                            last_hold_resend_wall = None
                            print(
                                "[mapping_server] Hold mode enabled. "
                                "Live history updates are paused; re-sending the latest output."
                            )
                        else:
                            print("[mapping_server] Hold mode already enabled.")
                        allow_fn = getattr(updater, "allow_control_enable", None)
                        if callable(allow_fn):
                            allow_fn(now=now)
                        updater.resend_current_output(now=now)
                        last_hold_resend_wall = now
                    elif key_lower == "d":
                        paused = True
                        manually_disabled = True
                        last_hold_resend_wall = None
                        print(
                            "[mapping_server] Manual disable enabled. "
                            "Live history updates are paused; disabling the active backend."
                        )
                        updater.disable_current_output(now=now)
                    elif key_lower == "a":
                        if paused or manually_disabled:
                            was_manually_disabled = manually_disabled
                            paused = False
                            manually_disabled = False
                            last_hold_resend_wall = None
                            print("[mapping_server] Live updates resumed.")
                            allow_fn = getattr(updater, "allow_control_enable", None)
                            if callable(allow_fn):
                                allow_fn(now=now)
                            if was_manually_disabled:
                                updater.resend_current_output(now=now)
                        else:
                            allow_fn = getattr(updater, "allow_control_enable", None)
                            if callable(allow_fn):
                                allow_fn(now=now)
                                print("[mapping_server] Live updates were already running; control enable is allowed.")
                            else:
                                print("[mapping_server] Live updates were already running.")

                now = time.perf_counter()
                control_event = _poll_control_command(
                    control_sock,
                    status_payload=_mapping_server_status_payload(
                        updater=updater,
                        args=args,
                        paused=paused,
                        manually_disabled=manually_disabled,
                        now=now,
                    ),
                )
                window = client.poll_window(timeout_ms=int(args.poll_timeout_ms))
                reset_event = client.pop_last_reset()
                now = time.perf_counter()
                if control_event is not None:
                    control_cmd = str(control_event.get("cmd", "")).strip().lower()
                    if control_cmd == "reset":
                        _apply_control_enable_delay(updater, control_event, now=now)
                        updater.process_once(
                            window=None,
                            reset_event=control_event,
                            now=now,
                        )
                        try:
                            updater.prepare_remote_controller()
                        except AdaptorBinUploadError as exc:
                            updater._record_permanent_prepare_failure(exc)
                            print(
                                "[mapping_server] Direct reset handled locally, but "
                                "controller rejected adaptor prepare: "
                                f"{exc}"
                            )
                        except Exception as exc:
                            updater.pending_remote_prepare = True
                            updater.last_prepare_attempt_wall = None
                            print(
                                "[mapping_server] Direct reset handled locally, but "
                                "controller re-prepare failed; will retry on history: "
                                f"{exc}"
                            )
                    elif control_cmd == "disable":
                        paused = True
                        manually_disabled = True
                        last_hold_resend_wall = None
                        print(
                            "[mapping_server] Control disable received. "
                            "Live history updates are paused; disabling the active backend."
                        )
                        updater.disable_current_output(now=now)
                    elif control_cmd == "hold":
                        paused = True
                        manually_disabled = False
                        last_hold_resend_wall = now
                        print(
                            "[mapping_server] Control hold received. "
                            "Live history updates are paused; re-sending the latest output."
                        )
                        _apply_control_enable_delay(updater, control_event, now=now)
                        updater.resend_current_output(now=now)
                    elif control_cmd == "resume":
                        was_manually_disabled = manually_disabled
                        paused = False
                        manually_disabled = False
                        last_hold_resend_wall = None
                        print("[mapping_server] Control resume received. Live updates resumed.")
                        _apply_control_enable_delay(updater, control_event, now=now)
                        if was_manually_disabled:
                            updater.resend_current_output(now=now)
                if paused:
                    updater.process_once(window=None, reset_event=reset_event, now=now)
                    if (
                        not manually_disabled
                        and (
                        last_hold_resend_wall is None
                        or (now - last_hold_resend_wall) >= _HOLD_RESEND_INTERVAL_S
                        )
                    ):
                        updater.resend_current_output(now=now)
                        last_hold_resend_wall = now
                else:
                    updater.process_once(window=window, reset_event=reset_event, now=now)
                    if bool(args.enable_viser) and window:
                        latest = window[-1]
                        if isinstance(latest, dict):
                            client.visualize_sample(latest)

                if not window:
                    if bool(args.debug) and (now - last_status_wall) >= 1.0:
                        if paused and manually_disabled:
                            wait_suffix = " (disabled)"
                        elif paused:
                            wait_suffix = " (hold mode)"
                        else:
                            wait_suffix = ""
                        print(f"[mapping_server] Waiting for history...{wait_suffix}")
                        last_status_wall = now
                    elif float(args.status_interval_s) > 0.0 and (
                        now - last_status_wall
                    ) >= float(args.status_interval_s):
                        if paused and manually_disabled:
                            paused_suffix = " paused=disabled"
                        elif paused:
                            paused_suffix = " paused=hold"
                        else:
                            paused_suffix = ""
                        print(updater.status_line(now=now) + paused_suffix)
                        last_status_wall = now
                    continue

                if float(args.status_interval_s) > 0.0 and (
                    now - last_status_wall
                ) >= float(args.status_interval_s):
                    if paused and manually_disabled:
                        paused_suffix = " paused=disabled"
                    elif paused:
                        paused_suffix = " paused=hold"
                    else:
                        paused_suffix = ""
                    print(updater.status_line(now=now) + paused_suffix)
                    last_status_wall = now
    except KeyboardInterrupt:
        print("\n[mapping_server] Exiting.")
    finally:
        try:
            updater.close()
        finally:
            if control_sock is not None:
                control_sock.close(linger=0)
            client.close()
