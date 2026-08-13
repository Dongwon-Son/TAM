from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import functools
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from simadaptor.deploy.runtime_common import (
    HistoryRuntimeBundle,
    mask_history_inputs_by_keep_mask,
    prepare_history_inputs,
)
from simadaptor.deploy.jax_cache import (
    DEFAULT_DEPLOY_JAX_CACHE_XLA_CACHES as DEFAULT_HISTORY_JAX_CACHE_XLA_CACHES,
    configure_jax_persistent_cache,
)
from simadaptor.eval.online_runtime import (
    attention_history_tokens_from_seconds,
    limit_decode_cache_attention_window,
)


DEFAULT_SIMADAPTOR_CKPT_PATH = "checkpoints/tam/example"
DEFAULT_DEPLOY_ATTENTION_HISTORY_S = 4.0


def _numeric_leaf_to_jax(value: Any) -> tuple[Any, bool]:
    if value is None:
        return None, True
    if isinstance(value, (str, bytes)):
        return value, False
    try:
        arr = np.asarray(value)
    except Exception:
        return value, False
    if arr.dtype.kind not in ("b", "i", "u", "f", "c"):
        return value, False
    return jnp.asarray(arr), True


def _tree_to_jax_dynamic_arg(tree: Any) -> tuple[Any, bool]:
    """Convert an array-like pytree to JAX arrays, reporting if every leaf is usable."""
    if tree is None:
        return None, True
    if is_dataclass(tree) and not isinstance(tree, type):
        tree = {field.name: getattr(tree, field.name) for field in fields(tree)}
    if isinstance(tree, Mapping):
        converted = {}
        ok = True
        for key, value in tree.items():
            converted_value, value_ok = _tree_to_jax_dynamic_arg(value)
            converted[key] = converted_value
            ok = ok and value_ok
        return converted, ok
    if isinstance(tree, tuple):
        converted_values = []
        ok = True
        for value in tree:
            converted_value, value_ok = _tree_to_jax_dynamic_arg(value)
            converted_values.append(converted_value)
            ok = ok and value_ok
        if hasattr(tree, "_fields"):
            return type(tree)(*converted_values), ok
        return tuple(converted_values), ok
    if isinstance(tree, list):
        converted_values = []
        ok = True
        for value in tree:
            converted_value, value_ok = _tree_to_jax_dynamic_arg(value)
            converted_values.append(converted_value)
            ok = ok and value_ok
        return converted_values, ok
    return _numeric_leaf_to_jax(tree)


def _bundle_from_sim_inf(sim_inf: Any) -> HistoryRuntimeBundle:
    if hasattr(sim_inf, "get_history_runtime_bundle"):
        bundle = sim_inf.get_history_runtime_bundle()
        if not isinstance(bundle, HistoryRuntimeBundle):
            raise TypeError(
                "SimAdaptorInference.get_history_runtime_bundle() must return "
                f"HistoryRuntimeBundle, got {type(bundle)!r}."
            )
        return bundle

    hist_models = getattr(sim_inf, "_simadaptor_model", None)
    simadaptor_params = getattr(sim_inf, "_simadaptor_params", None)
    if hist_models is None or simadaptor_params is None:
        raise TypeError(
            "RealTimeHistoryAdaptor requires either runtime_bundle=... or a "
            "sim_inf object that exposes get_history_runtime_bundle()."
        )
    return HistoryRuntimeBundle(
        hist_model=hist_models[0],
        hist_params=simadaptor_params["hist"],
        norm_stats=getattr(sim_inf, "_norm_stats", None),
        mjx_model_template=getattr(sim_inf, "mjx_model_template"),
        ideal_model_has_gravity=bool(getattr(sim_inf, "ideal_model_has_gravity")),
    )


class RealTimeHistoryAdaptor:
    """
    Minimal helper for real experiments that stream (q, qd, tau) and need the
    history embedding updated online.

    Assumptions:
    - Incoming samples are at the control rate used in training (dt that matches
      the saved model).
    - Tokens are windows of length ``patch_size`` (e.g., 400 samples by default).
      A new token is emitted every ``patch_stride`` samples (default 200) using
      a sliding window over the most recent ``patch_size`` samples.
    """

    def __init__(
        self,
        simadaptor_ckpt_path: str | None = None,
        *,
        sim_inf: Any | None = None,
        runtime_bundle: HistoryRuntimeBundle | None = None,
        checkpoint_step: int | None = None,
        xml_path: str | Path | None = None,
        expected_dt: float = 0.001,
        attention_history_s: float | None = DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
        jax_cache_dir: str | Path | None = None,
        jax_cache_min_compile_time_s: float = 0.0,
        jax_cache_min_entry_size_bytes: int = -1,
        jax_cache_xla_caches: str | None = DEFAULT_HISTORY_JAX_CACHE_XLA_CACHES,
    ):
        import simadaptor.models.transformer as models_transformer

        self._jax_cache_dir = configure_jax_persistent_cache(
            jax_cache_dir,
            min_compile_time_s=float(jax_cache_min_compile_time_s),
            min_entry_size_bytes=int(jax_cache_min_entry_size_bytes),
            xla_caches=jax_cache_xla_caches,
            log_prefix="[RealTimeHistoryAdaptor]",
        )
        self._models_transformer = models_transformer
        self._inf = sim_inf
        if runtime_bundle is None:
            if sim_inf is None:
                from simadaptor.deploy.inf_util import SimAdaptorInference

                self._inf = SimAdaptorInference(
                    simadaptor_ckpt_path=simadaptor_ckpt_path,
                    checkpoint_step=checkpoint_step,
                    xml_path=xml_path,
                )
            runtime_bundle = _bundle_from_sim_inf(self._inf)
        self._runtime_bundle = runtime_bundle

        self._dof = int(
            getattr(self._runtime_bundle.mjx_model_template, "nu", 0)
            or getattr(self._runtime_bundle.mjx_model_template, "nv", 0)
            or getattr(self._runtime_bundle.mjx_model_template, "nq", 0)
        )
        if self._dof <= 0:
            raise ValueError(
                "Failed to infer robot DoF for RealTimeHistoryAdaptor from the resolved XML/model."
            )
        hist_model = self._runtime_bundle.hist_model
        hist_cfg = getattr(hist_model, "cfg", hist_model)
        self._patch_size = int(getattr(hist_cfg, "patch_size"))
        self._patch_stride = int(getattr(hist_cfg, "patch_stride"))
        self._masked_fit_half = int(
            getattr(hist_cfg, "masked_fit_max_neighbors_each_side", 50) or 50
        )
        self._expected_dt = expected_dt
        if float(self._expected_dt) <= 0.0:
            raise ValueError(f"expected_dt must be positive, got {expected_dt}")

        self._history_smoothing = "masked_local_fit"
        self._context_half = self._masked_fit_half
        self._decode_patch_size = (
            int(self._patch_size + 2 * self._context_half)
            if self._context_half > 0
            else int(self._patch_size)
        )
        # The public release ships the jointwise_flat AR encoder with RoPE
        # locked on, so these runtime facts are fixed rather than read from
        # pruned config fields.
        self._jointwise_history = True
        self._cache_tokens_per_history_step = self._dof
        self._use_rope = True
        self._rope_base = float(getattr(hist_cfg, "rope_base", 10000.0))
        if attention_history_s is not None and float(attention_history_s) < 0.0:
            raise ValueError(
                "attention_history_s must be non-negative when set; "
                f"got {attention_history_s}."
            )
        self._attention_history_s = attention_history_s
        self._attention_history_tokens = attention_history_tokens_from_seconds(
            attention_history_s,
            self._expected_dt,
            decode_patch_size=self._decode_patch_size,
            patch_stride=self._patch_stride,
        )
        self._attention_history_cache_tokens = (
            None
            if self._attention_history_tokens is None
            else int(self._attention_history_tokens) * self._cache_tokens_per_history_step
        )
        self._hist_params_jit, self._hist_params_dynamic_arg = _tree_to_jax_dynamic_arg(
            self._runtime_bundle.hist_params
        )
        self._norm_stats_jit, self._norm_stats_dynamic_arg = _tree_to_jax_dynamic_arg(
            self._runtime_bundle.norm_stats
        )
        if not self._hist_params_dynamic_arg:
            self._hist_params_jit = None
        if not self._norm_stats_dynamic_arg:
            self._norm_stats_jit = None
        if not self._hist_params_dynamic_arg:
            print(
                "[RealTimeHistoryAdaptor] Warning: history params include non-array "
                "leaves; falling back to closed-over params for the decode JIT."
            )
        if not self._norm_stats_dynamic_arg:
            print(
                "[RealTimeHistoryAdaptor] Warning: norm stats include non-array "
                "leaves; falling back to closed-over norm_stats for the decode JIT."
            )

        self._cache = self._init_cache(batch_size=1)
        self._decode_step = self._build_decode_step()
        if self._context_half > 0:
            print(
                f"[RealTimeHistoryAdaptor] Masked local-fit smoothing context enabled: "
                f"decode_patch_size={self._decode_patch_size} "
                f"(patch_size={self._patch_size}, context_half={self._context_half})."
            )
        if self._attention_history_tokens is not None:
            print(
                "[RealTimeHistoryAdaptor] Transformer attention history limited: "
                f"attention_history_s={float(attention_history_s):g}, "
                f"temporal_history_patches={self._attention_history_tokens}, "
                f"cache_tokens={self._attention_history_cache_tokens}, "
                f"tokens_per_history_step={self._cache_tokens_per_history_step}."
            )
        print("Warming up JIT for RealTimeHistoryAdaptor...")
        warmup_start_time = time.time()
        self._warm_decode_jit()
        print(f"JIT warm-up complete. Time taken: {time.time() - warmup_start_time:.2f} seconds.")

        self.history_emb = None  # shape [1, emb_dim] or [1, DoF, emb_dim]

        self._ts_buf = np.zeros((0,), dtype=np.float64)
        self._idx_buf = np.zeros((0,), dtype=np.int64)
        self._q_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._qd_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._u_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._keep_buf = np.zeros((0,), dtype=np.float32)

        self._t0 = None
        self._base_dt = None
        self._next_emit_idx = None

    @property
    def inf(self):
        return self._inf

    @property
    def runtime_bundle(self) -> HistoryRuntimeBundle:
        return self._runtime_bundle

    def reset(self) -> None:
        """Reset streaming buffers and decoder cache (e.g., after a controller reset)."""
        self._cache = self._init_cache(batch_size=1)
        self.history_emb = None

        self._ts_buf = np.zeros((0,), dtype=np.float64)
        self._idx_buf = np.zeros((0,), dtype=np.int64)
        self._q_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._qd_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._u_buf = np.zeros((0, self._dof), dtype=np.float32)
        self._keep_buf = np.zeros((0,), dtype=np.float32)

        self._t0 = None
        self._base_dt = None
        self._next_emit_idx = None

    def _init_cache(self, batch_size: int):
        return self._models_transformer.init_infer_state(
            self._runtime_bundle.hist_params,
            self._runtime_bundle.hist_model,
            batch_size=batch_size,
        )

    def _build_decode_step(self):
        """
        Build a jitted single-token decode to keep shapes static and compilation
        overhead minimal. Inputs must be [1,1,P,DoF] where P is the decode patch size.
        The output preserves any joint axis from joint-wise checkpoints.
        """
        closed_params = self._runtime_bundle.hist_params
        model = self._runtime_bundle.hist_model
        closed_norm_stats = self._runtime_bundle.norm_stats
        use_dynamic_params = bool(self._hist_params_dynamic_arg)
        use_dynamic_norm_stats = bool(self._norm_stats_dynamic_arg)
        valid_mask = jnp.ones((1, 1), dtype=jnp.float32)
        attention_history_cache_tokens = self._attention_history_cache_tokens
        cache_tokens_per_history_step = self._cache_tokens_per_history_step
        use_rope = self._use_rope
        rope_base = self._rope_base

        @functools.partial(jax.jit, static_argnames=())
        def _step(
            params_arg,
            norm_stats_arg,
            cache,
            q_patch,
            qd_patch,
            u_patch,
            input_keep_mask,
        ):
            cache = limit_decode_cache_attention_window(
                cache,
                attention_history_cache_tokens,
                append_tokens=cache_tokens_per_history_step,
                rebase_rope=use_rope,
                rope_base=rope_base,
            )
            emb, cache_out = self._models_transformer.step_decode(
                params=params_arg if use_dynamic_params else closed_params,
                cache=cache,
                model=model,
                chunk_q=q_patch,
                chunk_qd=qd_patch,
                chunk_u=u_patch,
                valid_mask=valid_mask,
                key=None,
                norm_stats=norm_stats_arg if use_dynamic_norm_stats else closed_norm_stats,
                input_keep_mask=input_keep_mask,
            )
            return emb[:, 0, ...], cache_out

        return _step

    def _warm_decode_jit(self):
        """Compile the decode step once at init to avoid runtime JIT pauses."""
        dummy = jnp.zeros((1, 1, self._decode_patch_size, self._dof), dtype=jnp.float32)
        dummy_keep = jnp.ones((1, 1, self._decode_patch_size), dtype=jnp.float32)
        cache0 = self._cache
        emb0, cache1 = self._decode_step(
            self._hist_params_jit,
            self._norm_stats_jit,
            cache0,
            dummy,
            dummy,
            dummy,
            dummy_keep,
        )
        jax.block_until_ready(emb0)
        emb1, _ = self._decode_step(
            self._hist_params_jit,
            self._norm_stats_jit,
            cache1,
            dummy,
            dummy,
            dummy,
            dummy_keep,
        )
        jax.block_until_ready(emb1)
        self._cache = self._init_cache(batch_size=1)

    def push_sample(
        self,
        q,
        qd,
        tau,
        gravity=None,
        *,
        tau_is_model_space: bool = False,
        raw_tau=None,
        keep_mask=None,
    ) -> jnp.ndarray | None:
        """
        Push one timestep (numpy arrays). Returns the latest history embedding
        once enough samples have been accumulated; otherwise returns None.
        """
        ts = np.asarray(
            self._ts_buf[-1] + self._expected_dt if self._ts_buf.size else 0.0,
            dtype=np.float64,
        )
        keep = None if keep_mask is None else np.asarray([keep_mask], dtype=np.float32)
        return self.push_window(
            timestamps=np.asarray([ts]),
            q=np.asarray([q], dtype=np.float32),
            qd=np.asarray([qd], dtype=np.float32),
            tau=np.asarray([tau], dtype=np.float32),
            gravity=None if gravity is None else np.asarray([gravity], dtype=np.float32),
            tau_is_model_space=tau_is_model_space,
            raw_tau=None if raw_tau is None else np.asarray([raw_tau], dtype=np.float32),
            keep_mask=keep,
        )

    def push_window(
        self,
        timestamps,
        q,
        qd,
        tau,
        gravity=None,
        *,
        tau_is_model_space: bool = False,
        raw_tau=None,
        keep_mask=None,
    ) -> jnp.ndarray | None:
        """
        Push a window of samples: timestamps [N], q/qd/tau [N, dof].
        By default `tau` is interpreted as raw controller torque history
        (`tau_cmd`). When the loaded checkpoint has `ideal_model_has_gravity=True`,
        pass the matching logged gravity torque via `gravity` so this helper can
        build the model-space torque history expected by the history encoder.
        If `tau_is_model_space=True`, `tau` is assumed to already be in model
        space and `gravity` is ignored.
        If `raw_tau` is provided, it is used only as the zero-torque masking
        reference; this is useful for fused histories where a base or residual
        torque stream should share the applied-torque validity mask.
        Raw zero-torque rows and explicit keep-mask invalid rows are zeroed
        before buffering, matching the deploy adaptor's history semantics.
        - Handles variable receive window sizes/frequencies.
        - Deduplicates overlapping data using timestamps.
        - Reconstructs a dense runtime buffer on the expected sample grid, with
          missing slots zero padded and marked invalid.
        - Emits tokens every patch_stride samples; returns the latest embedding
          if one or more tokens were generated, else None.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        q = np.asarray(q, dtype=np.float32)
        qd = np.asarray(qd, dtype=np.float32)
        tau = np.asarray(tau, dtype=np.float32)
        n_new = timestamps.shape[0]
        if n_new == 0:
            return None
        if q.ndim != 2 or q.shape[1] != self._dof:
            raise ValueError(f"q must have shape [N, {self._dof}], got {q.shape}")
        if qd.ndim != 2 or qd.shape[1] != self._dof:
            raise ValueError(f"qd must have shape [N, {self._dof}], got {qd.shape}")
        if tau.ndim != 2 or tau.shape[1] != self._dof:
            raise ValueError(f"tau must have shape [N, {self._dof}], got {tau.shape}")

        q, qd, tau_model, keep_new = prepare_history_inputs(
            q,
            qd,
            tau,
            gravity=gravity,
            ideal_model_has_gravity=self._runtime_bundle.ideal_model_has_gravity,
            context="RealTimeHistoryAdaptor.push_window",
            tau_is_model_space=tau_is_model_space,
            apply_zero_torque_mask=True,
            raw_tau=raw_tau,
        )
        if keep_mask is not None:
            keep_external = np.asarray(keep_mask, dtype=np.float32).reshape(-1)
            if keep_external.shape != keep_new.shape:
                raise ValueError(
                    f"keep_mask must match the leading timestep shape {keep_new.shape}, got {keep_external.shape}"
                )
            keep_new = keep_new * keep_external
            q, qd, tau_model = mask_history_inputs_by_keep_mask(q, qd, tau_model, keep_new)

        if self._t0 is None:
            self._t0 = float(timestamps[0])
        self._base_dt = float(self._expected_dt)

        idx_new = np.rint((timestamps - self._t0) / self._base_dt).astype(np.int64)
        idx_new = np.maximum.accumulate(idx_new)

        idx_all = np.concatenate([self._idx_buf, idx_new], axis=0)
        ts_all = np.concatenate([self._ts_buf, timestamps], axis=0)
        q_all = np.concatenate([self._q_buf, q], axis=0)
        qd_all = np.concatenate([self._qd_buf, qd], axis=0)
        tau_all = np.concatenate([self._u_buf, tau_model], axis=0)
        keep_all = np.concatenate([self._keep_buf, keep_new], axis=0)

        order = np.argsort(idx_all, kind="mergesort")
        idx_sorted = idx_all[order]
        ts_sorted = ts_all[order]
        q_sorted = q_all[order]
        qd_sorted = qd_all[order]
        tau_sorted = tau_all[order]
        keep_sorted = keep_all[order]

        keep_mask_unique = np.concatenate(
            [idx_sorted[1:] != idx_sorted[:-1], np.array([True])], axis=0
        )
        idx_sorted = idx_sorted[keep_mask_unique]
        ts_sorted = ts_sorted[keep_mask_unique]
        q_sorted = q_sorted[keep_mask_unique]
        qd_sorted = qd_sorted[keep_mask_unique]
        tau_sorted = tau_sorted[keep_mask_unique]
        keep_sorted = keep_sorted[keep_mask_unique]

        self._idx_buf, self._ts_buf, self._q_buf, self._qd_buf, self._u_buf, self._keep_buf = (
            idx_sorted,
            ts_sorted,
            q_sorted,
            qd_sorted,
            tau_sorted,
            keep_sorted,
        )

        if self._next_emit_idx is None and self._idx_buf.size > 0:
            start_idx = int(self._idx_buf[0])
            # The first decodable patch is delayed until local-fit smoothing has
            # both left and right context around the central patch window.
            self._next_emit_idx = start_idx + self._patch_size - 1 + self._context_half

        if self._idx_buf.size > 0:
            dense_start = int(self._idx_buf[0])
            dense_end = int(self._idx_buf[-1])
            dense_idx = np.arange(dense_start, dense_end + 1, dtype=np.int64)
            dense_q = np.zeros((dense_idx.shape[0], self._dof), dtype=np.float32)
            dense_qd = np.zeros_like(dense_q)
            dense_u = np.zeros_like(dense_q)
            dense_keep = np.zeros((dense_idx.shape[0],), dtype=np.float32)
            dense_ts = self._t0 + dense_idx.astype(np.float64) * self._base_dt
            offsets = (self._idx_buf - dense_start).astype(np.int64)
            dense_q[offsets] = self._q_buf
            dense_qd[offsets] = self._qd_buf
            dense_u[offsets] = self._u_buf
            dense_keep[offsets] = self._keep_buf
            dense_ts[offsets] = self._ts_buf
            self._idx_buf = dense_idx
            self._ts_buf = dense_ts
            self._q_buf = dense_q
            self._qd_buf = dense_qd
            self._u_buf = dense_u
            self._keep_buf = dense_keep

        def _dense_patch(end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
            if self._idx_buf.size == 0:
                return None
            end_ext = int(end_idx) + int(self._context_half)
            start_ext = end_ext - int(self._decode_patch_size) + 1
            if start_ext < int(self._idx_buf[0]) or end_ext > int(self._idx_buf[-1]):
                return None
            offset_start = start_ext - int(self._idx_buf[0])
            offset_end = end_ext - int(self._idx_buf[0]) + 1
            if (offset_end - offset_start) != self._decode_patch_size:
                return None
            return (
                np.asarray(self._q_buf[offset_start:offset_end], dtype=np.float32),
                np.asarray(self._qd_buf[offset_start:offset_end], dtype=np.float32),
                np.asarray(self._u_buf[offset_start:offset_end], dtype=np.float32),
                np.asarray(self._keep_buf[offset_start:offset_end], dtype=np.float32),
            )

        latest_emb = None
        while (
            self._next_emit_idx is not None
            and self._idx_buf.size > 0
            and self._idx_buf[-1] >= (self._next_emit_idx + self._context_half)
        ):
            patch = _dense_patch(int(self._next_emit_idx))
            if patch is None:
                break
            q_np, qd_np, u_np, keep_np = patch
            if q_np.shape[0] != self._decode_patch_size:
                break

            q_patch = jnp.asarray(q_np, dtype=jnp.float32)[None, None, ...]
            qd_patch = jnp.asarray(qd_np, dtype=jnp.float32)[None, None, ...]
            u_patch = jnp.asarray(u_np, dtype=jnp.float32)[None, None, ...]
            input_keep_mask = jnp.asarray(keep_np, dtype=jnp.float32)[None, None, ...]

            latest_emb, self._cache = self._decode_step(
                self._hist_params_jit,
                self._norm_stats_jit,
                self._cache,
                q_patch,
                qd_patch,
                u_patch,
                input_keep_mask,
            )
            self.history_emb = latest_emb
            self._next_emit_idx += self._patch_stride

        max_keep = self._decode_patch_size + 2 * self._patch_stride
        if self._idx_buf.shape[0] > max_keep:
            self._idx_buf = self._idx_buf[-max_keep:]
            self._ts_buf = self._ts_buf[-max_keep:]
            self._q_buf = self._q_buf[-max_keep:]
            self._qd_buf = self._qd_buf[-max_keep:]
            self._u_buf = self._u_buf[-max_keep:]
            self._keep_buf = self._keep_buf[-max_keep:]

        return latest_emb


def _load_validation_traj(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    def _pick(keys: list[str], label: str) -> np.ndarray:
        for key in keys:
            if key in data:
                return np.asarray(data[key])
        raise KeyError(f"Missing {label} in {npz_path}; tried keys={keys}")

    q = _pick(["q"], "q")
    qd = _pick(["qd", "dq"], "qd")
    tau = _pick(["tau_cmd", "tau_commanded", "u_des", "tau"], "tau")
    gravity = np.asarray(data["gravity"]) if "gravity" in data else None
    t = _pick(["t", "times", "t_raw"], "t")
    return q, qd, tau, gravity, t


def _is_uniform_dt(t: np.ndarray, expected_dt: float, tol_ratio: float = 0.01) -> bool:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return True
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)):
        return False
    med = float(np.median(dt))
    if abs(med - expected_dt) > expected_dt * tol_ratio:
        return False
    return float(np.std(dt)) <= expected_dt * tol_ratio


def _uniform_times(n_samples: int, dt: float, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(int(n_samples), dtype=np.float64) * float(dt)


def _reflect_pad_tail(x: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return x[:0]
    x = np.asarray(x)
    n = x.shape[0]
    if n == 0:
        raise ValueError("Cannot pad an empty sequence.")
    if n == 1:
        return np.repeat(x, pad, axis=0)
    idx = np.arange(n, n + pad, dtype=np.int64)
    period = 2 * n - 2
    idx0 = idx % period
    idx1 = np.where(idx0 <= (n - 1), idx0, period - idx0)
    return x[idx1]


def _token_end_indices(n_samples: int, patch_size: int, patch_stride: int) -> np.ndarray:
    if n_samples < patch_size:
        return np.zeros((0,), dtype=np.int64)
    n_tokens = 1 + int(np.floor((n_samples - patch_size) / patch_stride))
    return (patch_size - 1) + np.arange(n_tokens, dtype=np.int64) * int(patch_stride)


def _trim_to_stride(
    q: np.ndarray,
    qd: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    patch_size: int,
    patch_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if q.shape[0] < patch_size:
        return q, qd, u, t
    n_windows = 1 + int(np.floor((q.shape[0] - patch_size) / patch_stride))
    trim_len = patch_size + (n_windows - 1) * patch_stride
    start = max(0, q.shape[0] - trim_len)
    print(f"[trim_to_stride] trimming trajectory from {q.shape[0]} to {trim_len} samples (start={start})")
    if start == 0:
        return q, qd, u, t
    return q[start:], qd[start:], u[start:], t[start:]


def validate_online_vs_static(
    *,
    traj_npz: str | Path,
    simadaptor_ckpt_path: str | None = None,
    xml_path: str | Path | None = None,
    expected_dt: float = 0.001,
    attention_history_s: float | None = DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
    stream_window: int | None = None,
    seed: int = 0,
    trim_to_stride: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    import simadaptor.physics.smoothing as smoothing_util

    adaptor = RealTimeHistoryAdaptor(
        simadaptor_ckpt_path=simadaptor_ckpt_path,
        xml_path=xml_path,
        expected_dt=expected_dt,
        attention_history_s=attention_history_s,
    )
    sim_inf = adaptor.inf
    if sim_inf is None:
        raise RuntimeError("validate_online_vs_static requires an attached SimAdaptorInference.")

    q_raw, qd_raw, tau_raw, gravity_raw, t_raw = _load_validation_traj(traj_npz)
    t_raw = np.asarray(t_raw, dtype=np.float64).reshape(-1)
    if _is_uniform_dt(t_raw, expected_dt):
        q_u = np.asarray(q_raw, dtype=np.float32)
        qd_u = np.asarray(qd_raw, dtype=np.float32)
        tau_u_raw = np.asarray(tau_raw, dtype=np.float32)
        gravity_u = None if gravity_raw is None else np.asarray(gravity_raw, dtype=np.float32)
    else:
        q_u, qd_u, _, tau_u_raw, _ = smoothing_util.q_traj_to_traj(
            q_raw, tau_raw, t_raw, dt=expected_dt
        )
        if sim_inf.ideal_model_has_gravity and gravity_raw is not None:
            _, _, _, gravity_u, _ = smoothing_util.q_traj_to_traj(
                q_raw, gravity_raw, t_raw, dt=expected_dt
            )
        else:
            gravity_u = None
        q_u = np.asarray(q_u, dtype=np.float32)
        qd_u = np.asarray(qd_u, dtype=np.float32)
        tau_u_raw = np.asarray(tau_u_raw, dtype=np.float32)
        gravity_u = None if gravity_u is None else np.asarray(gravity_u, dtype=np.float32)

    q_u, qd_u, u_u, keep_u = prepare_history_inputs(
        q_u,
        qd_u,
        tau_u_raw,
        gravity=gravity_u,
        ideal_model_has_gravity=sim_inf.ideal_model_has_gravity,
        context=f"validate_online_vs_static({traj_npz})",
        apply_zero_torque_mask=True,
    )
    t_u = _uniform_times(q_u.shape[0], expected_dt, t0=0.0)

    if trim_to_stride:
        q_u, qd_u, u_u, t_u = _trim_to_stride(
            q_u, qd_u, u_u, t_u, adaptor._patch_size, adaptor._patch_stride
        )
        keep_u = keep_u[-q_u.shape[0] :]

    rng = jax.random.PRNGKey(seed)
    emb_static = sim_inf.history_encoding(
        q_u,
        qd_u,
        u_u,
        rng,
        input_keep_mask=keep_u[None, :],
        tau_is_model_space=True,
    )
    emb_static = np.asarray(emb_static)
    static_end = _token_end_indices(q_u.shape[0], adaptor._patch_size, adaptor._patch_stride)

    adaptor.reset()
    if stream_window is None:
        stream_window = adaptor._patch_stride
    pad_online = adaptor._context_half if adaptor._context_half > 0 else 0
    if pad_online > 0:
        q_online = np.concatenate([q_u, _reflect_pad_tail(q_u, pad_online)], axis=0)
        qd_online = np.concatenate([qd_u, _reflect_pad_tail(qd_u, pad_online)], axis=0)
        u_online = np.concatenate([u_u, _reflect_pad_tail(u_u, pad_online)], axis=0)
    else:
        q_online, qd_online, u_online = q_u, qd_u, u_u
    t_online = _uniform_times(q_online.shape[0], expected_dt, t0=0.0)

    online_tokens = []
    for start in range(0, t_online.shape[0], stream_window):
        end = min(t_online.shape[0], start + stream_window)
        emb = adaptor.push_window(
            timestamps=t_online[start:end],
            q=q_online[start:end],
            qd=qd_online[start:end],
            tau=u_online[start:end],
            tau_is_model_space=True,
        )
        if emb is not None:
            online_tokens.append(np.asarray(emb)[0])

    if not online_tokens:
        raise RuntimeError("No online embeddings produced; trajectory may be too short for patch_size.")
    emb_online = np.stack(online_tokens, axis=0)
    online_end = (adaptor._patch_size - 1) + np.arange(
        emb_online.shape[0], dtype=np.int64
    ) * int(adaptor._patch_stride)

    min_end = adaptor._patch_size - 1
    max_end = q_u.shape[0] - 1
    if adaptor._context_half > 0:
        min_end = max(min_end, adaptor._context_half + adaptor._patch_size - 1)
        max_end = min(max_end, q_u.shape[0] - 1 - adaptor._context_half)

    static_mask = (static_end >= min_end) & (static_end <= max_end)
    online_mask = (online_end >= min_end) & (online_end <= max_end) & (online_end <= (q_u.shape[0] - 1))
    static_end = static_end[static_mask]
    online_end = online_end[online_mask]
    common_end = np.intersect1d(static_end, online_end)
    if common_end.size == 0:
        raise RuntimeError("No overlapping tokens after boundary alignment.")

    common_set = {int(e) for e in common_end}
    static_lookup = {
        int(e): emb_static[i]
        for i, e in enumerate(_token_end_indices(q_u.shape[0], adaptor._patch_size, adaptor._patch_stride))
        if int(e) in common_set
    }
    online_lookup = {
        int(e): emb_online[i]
        for i, e in enumerate(
            (adaptor._patch_size - 1)
            + np.arange(emb_online.shape[0], dtype=np.int64) * int(adaptor._patch_stride)
        )
        if int(e) in common_set
    }
    emb_static = np.stack([static_lookup[int(e)] for e in common_end], axis=0)
    emb_online = np.stack([online_lookup[int(e)] for e in common_end], axis=0)

    diff = emb_online - emb_static
    abs_diff = np.abs(diff)
    l2 = np.linalg.norm(diff, axis=-1)
    last_abs_mean = float(abs_diff[-1].mean()) if abs_diff.shape[0] else float("nan")

    print(
        "[validate] online vs static history embedding",
        f"tokens={emb_online.shape[0]}, dim={emb_online.shape[1]}",
        f"aligned_end_idx=[{int(min_end)}, {int(max_end)}]",
        f"abs_mean={float(abs_diff.mean()):.6g}",
        f"abs_max={float(abs_diff.max()):.6g}",
        f"l2_mean={float(l2.mean()):.6g}",
        f"l2_max={float(l2.max()):.6g}",
        f"last_abs_mean={last_abs_mean:.6g}",
    )
    return emb_online, emb_static


def example_usage():
    """
    Sketch of how to integrate with a ZMQ/Numpy stream.
    Replace the fake loop with your recv logic (timestamps, q, qd, tau per window).
    """
    simadaptor_ckpt_path = DEFAULT_SIMADAPTOR_CKPT_PATH
    adaptor = RealTimeHistoryAdaptor(
        simadaptor_ckpt_path=simadaptor_ckpt_path,
        expected_dt=0.001,
    )

    recv_window = 200
    sample_dt = 0.001
    recv_dt = 0.01

    t0 = 0.0
    for i in tqdm(range(1000)):
        start_t = t0 + i * recv_dt
        timestamps = start_t + np.arange(recv_window) * sample_dt
        q = np.zeros((recv_window, adaptor._dof), dtype=np.float32)
        qd = np.zeros_like(q)
        tau = np.zeros_like(q)

        hist = adaptor.push_window(timestamps, q, qd, tau)
        if hist is not None:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SimAdaptor online history embedding utilities.")
    parser.add_argument("--mode", choices=["validate", "demo"], default="validate")
    parser.add_argument("--traj-npz", type=Path, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--expected-dt", type=float, default=0.001)
    parser.add_argument(
        "--attention-history-s",
        type=float,
        default=DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
        help=(
            "Limit transformer decode-cache attention to this many seconds "
            f"(default: {DEFAULT_DEPLOY_ATTENTION_HISTORY_S:g}); <=0 keeps the full cache."
        ),
    )
    parser.add_argument("--stream-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim-to-stride", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.mode == "demo":
        example_usage()
    else:
        if args.traj_npz is None or args.ckpt_path is None:
            raise SystemExit("validate mode requires --traj-npz and --ckpt-path.")
        validate_online_vs_static(
            traj_npz=args.traj_npz,
            simadaptor_ckpt_path=args.ckpt_path,
            expected_dt=float(args.expected_dt),
            attention_history_s=args.attention_history_s,
            stream_window=args.stream_window,
            seed=int(args.seed),
            trim_to_stride=bool(args.trim_to_stride),
        )


if __name__ == "__main__":
    main()
