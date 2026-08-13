from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from flax.struct import dataclass as flax_dataclass
import jax
import jax.numpy as jnp

import simadaptor.models.transformer as models_transformer


_NORM_STAT_DEFAULTS = {
    "mean_q": 0.0,
    "mean_dq": 0.0,
    "mean_u": 0.0,
    "var_q": 1.0,
    "var_dq": 1.0,
    "var_u": 1.0,
}


@flax_dataclass
class OnlineHistoryState:
    history_emb: jax.Array
    cache: Any
    q_buf: jax.Array
    qd_buf: jax.Array
    tau_buf: jax.Array
    keep_buf: jax.Array
    sample_count: jax.Array
    next_emit_idx: jax.Array
    has_embedding: jax.Array


@dataclass(frozen=True)
class OnlineHistoryRuntimeConfig:
    patch_size: int
    patch_stride: int
    context_half: int
    decode_patch_size: int
    # Historical name: this is a temporal-patch count and is passed to the
    # offline transformer's train_window_override.
    attention_history_tokens: int | None
    # Actual flattened KV-cache budget (temporal patches × DoF for jointwise).
    attention_history_cache_tokens: int | None
    cache_tokens_per_history_step: int
    use_rope: bool
    rope_base: float
    arm_dof: int
    emb_dim: int
    jointwise: bool


@dataclass(frozen=True)
class OnlineHistoryRuntime:
    config: OnlineHistoryRuntimeConfig
    cache0: Any
    decode_step: Callable[..., tuple[jax.Array, Any]]


def _resize_norm_stat_leaf(value: Any, dof: int, fill_value: float) -> Any:
    arr = jnp.asarray(value)
    if arr.ndim == 0 or int(arr.shape[-1]) == int(dof):
        return value
    if int(arr.shape[-1]) > int(dof):
        return arr[..., : int(dof)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, int(dof) - int(arr.shape[-1]))
    return jnp.pad(arr, tuple(pad_width), mode="constant", constant_values=fill_value)


def align_norm_stats_to_dof(norm_stats: Any, dof: int) -> Any:
    """Make saved per-DoF norm stats usable by narrower online eval controllers."""
    if norm_stats is None:
        return None
    dof = int(dof)
    if dof <= 0:
        return norm_stats
    if isinstance(norm_stats, Mapping):
        out = dict(norm_stats)
        for field, fill in _NORM_STAT_DEFAULTS.items():
            if field in out:
                out[field] = _resize_norm_stat_leaf(out[field], dof, fill)
        return out
    updates = {}
    for field, fill in _NORM_STAT_DEFAULTS.items():
        if hasattr(norm_stats, field):
            updates[field] = _resize_norm_stat_leaf(getattr(norm_stats, field), dof, fill)
    if not updates:
        return norm_stats
    if hasattr(norm_stats, "replace"):
        return norm_stats.replace(**updates)
    return norm_stats.__class__(**{**vars(norm_stats), **updates})


def push_window(window: jax.Array, new_val: jax.Array) -> jax.Array:
    new_val = jnp.asarray(new_val, dtype=window.dtype)
    return jnp.concatenate([window[1:], new_val[None, ...]], axis=0)


def attention_history_tokens_from_seconds(
    history_s: float | None,
    sample_dt_s: float | None,
    *,
    decode_patch_size: int,
    patch_stride: int,
) -> int | None:
    """Convert a raw-sample history horizon into temporal history patches."""
    if history_s is None or float(history_s) <= 0.0:
        return None
    if sample_dt_s is None or float(sample_dt_s) <= 0.0:
        raise ValueError("sample_dt_s must be positive when attention history is bounded.")
    decode_patch_size = max(int(decode_patch_size), 1)
    patch_stride = max(int(patch_stride), 1)
    history_samples = max(int(round(float(history_s) / float(sample_dt_s))), 1)
    if history_samples <= decode_patch_size:
        history_steps = 1
    else:
        history_steps = max(1, 1 + (history_samples - decode_patch_size) // patch_stride)
    return history_steps


def _cache_sequence_axis(arr: jax.Array) -> int:
    if arr.ndim < 3:
        raise ValueError(f"Expected a cached key/value with rank >= 3, got shape={arr.shape}.")
    return arr.ndim - 3


def _drop_oldest_cache_tokens(arr: jax.Array, count: int) -> jax.Array:
    seq_axis = _cache_sequence_axis(arr)
    seq_len = int(arr.shape[seq_axis])
    count = min(max(int(count), 1), seq_len)
    if count >= seq_len:
        return jnp.zeros_like(arr)
    tail = jnp.take(arr, jnp.arange(count, seq_len, dtype=jnp.int32), axis=seq_axis)
    pad = jnp.take(
        jnp.zeros_like(arr),
        jnp.arange(count, dtype=jnp.int32),
        axis=seq_axis,
    )
    return jnp.concatenate([tail, pad], axis=seq_axis)


def _rebase_rope_cached_key(
    cached_key: jax.Array,
    *,
    dropped_tokens: int,
    rope_base: float,
) -> jax.Array:
    """Move already-RoPE-rotated keys back by ``dropped_tokens`` positions."""

    head_dim = int(cached_key.shape[-1])
    if head_dim % 2:
        raise ValueError(f"RoPE cached-key head dimension must be even, got {head_dim}.")
    half = head_dim // 2
    inv_freq = 1.0 / (
        float(rope_base)
        ** (jnp.arange(half, dtype=jnp.float32) / max(half, 1))
    )
    angle = -float(dropped_tokens) * inv_freq
    cos = jnp.cos(angle).astype(cached_key.dtype)
    sin = jnp.sin(angle).astype(cached_key.dtype)
    even = cached_key[..., ::2]
    odd = cached_key[..., 1::2]
    rebased_even = even * cos - odd * sin
    rebased_odd = even * sin + odd * cos
    return jnp.reshape(
        jnp.stack([rebased_even, rebased_odd], axis=-1),
        cached_key.shape,
    )


def limit_decode_cache_attention_window(
    cache: Any,
    keep_tokens: int | None,
    *,
    append_tokens: int = 1,
    rebase_rope: bool = False,
    rope_base: float = 10000.0,
) -> Any:
    """Keep a fixed-size sliding token window in mutable Transformer decode caches.

    The cache shape stays unchanged for JIT compatibility. Once a cache leaf has
    insufficient room for the next decode chunk, that many oldest entries are
    shifted out and the chunk writes into the newly opened slots.  Jointwise
    models must pass ``append_tokens=DoF`` because one temporal patch appends one
    cache token per joint.

    RoPE keys are stored after positional rotation.  When ``rebase_rope`` is
    true, shifted keys are counter-rotated by the number of dropped positions so
    their relative positions remain correct after the sliding-window rebase.
    """
    if keep_tokens is None:
        return cache
    keep_tokens = int(keep_tokens)
    if keep_tokens <= 0:
        return cache
    append_tokens = max(int(append_tokens), 1)

    def visit(node):
        if not isinstance(node, Mapping):
            return node
        if {"cached_key", "cached_value", "cache_index"}.issubset(node.keys()):
            cached_key = node["cached_key"]
            cached_value = node["cached_value"]
            cache_index = node["cache_index"]
            capacity = int(cached_key.shape[_cache_sequence_axis(cached_key)])
            keep = max(keep_tokens, 1)
            if keep > capacity:
                raise ValueError(
                    f"Requested cache window {keep} exceeds cache capacity {capacity}."
                )
            if append_tokens > keep:
                raise ValueError(
                    f"append_tokens={append_tokens} exceeds keep_tokens={keep}."
                )
            if keep % append_tokens:
                raise ValueError(
                    "keep_tokens must be an integer number of decode chunks: "
                    f"keep_tokens={keep}, append_tokens={append_tokens}."
                )
            drop = append_tokens

            def shifted():
                out = dict(node)
                shifted_key = _drop_oldest_cache_tokens(cached_key, drop)
                # Current RoPE attention caches carry an absolute position
                # counter, so shifted keys keep their original rotation and
                # need no lossy repeated counter-rotation. Keep the fallback
                # for legacy/custom cache structures without that counter.
                if rebase_rope and "cache_rope_index" not in node:
                    shifted_key = _rebase_rope_cached_key(
                        shifted_key,
                        dropped_tokens=drop,
                        rope_base=rope_base,
                    )
                out["cached_key"] = shifted_key
                out["cached_value"] = _drop_oldest_cache_tokens(cached_value, drop)
                out["cache_index"] = jnp.maximum(
                    jnp.asarray(cache_index) - drop,
                    0,
                ).astype(jnp.asarray(cache_index).dtype)
                return out

            def unchanged():
                return dict(node)

            should_shift = jnp.all(jnp.asarray(cache_index) + append_tokens > keep)
            return jax.lax.cond(should_shift, shifted, unchanged)
        return {key: visit(value) for key, value in node.items()}

    return visit(cache)


def zero_torque_history_keep_mask(
    raw_tau: jax.Array,
    *,
    threshold: float = 1e-5,
) -> jax.Array:
    raw_tau = jnp.asarray(raw_tau)
    return jnp.where(
        jnp.all(jnp.abs(raw_tau) <= float(threshold), axis=-1),
        0.0,
        1.0,
    ).astype(raw_tau.dtype)


def mask_zero_torque_history_sample(
    q: jax.Array,
    qd: jax.Array,
    tau_model: jax.Array,
    raw_tau: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keep = zero_torque_history_keep_mask(raw_tau)
    return q * keep, qd * keep, tau_model * keep, keep


def runtime_history_embedding_from_sequence(history_seq: jax.Array) -> jax.Array:
    history_seq = jnp.asarray(history_seq)
    if history_seq.ndim == 1:
        return history_seq[None, :]
    if history_seq.ndim == 2:
        return history_seq[-1:, :]
    if history_seq.ndim == 3:
        return history_seq[-1:, :, :]
    if history_seq.ndim == 4:
        return history_seq[:, -1, ...]
    raise ValueError(
        f"Unexpected history embedding rank: {history_seq.ndim}, shape={history_seq.shape}"
    )


def build_online_history_runtime(
    *,
    hist_model,
    params_hist_example,
    emb_dim: int,
    arm_dof: int,
    attention_history_s: float | None = None,
    sample_dt_s: float | None = None,
) -> OnlineHistoryRuntime:
    hist_cfg = getattr(hist_model, "cfg", hist_model)
    patch_size = int(getattr(hist_cfg, "patch_size"))
    patch_stride = int(getattr(hist_cfg, "patch_stride"))
    masked_fit_half = int(
        getattr(hist_cfg, "masked_fit_max_neighbors_each_side", 50) or 0
    )
    context_half = masked_fit_half if masked_fit_half > 0 else 0
    decode_patch_size = patch_size + 2 * context_half if context_half > 0 else patch_size
    cache_tokens_per_history_step = int(arm_dof)
    use_rope = bool(getattr(hist_cfg, "use_RoPE", True))
    rope_base = float(getattr(hist_cfg, "rope_base", 10000.0))
    attention_history_tokens = attention_history_tokens_from_seconds(
        attention_history_s,
        sample_dt_s,
        decode_patch_size=decode_patch_size,
        patch_stride=patch_stride,
    )
    attention_history_cache_tokens = (
        None
        if attention_history_tokens is None
        else int(attention_history_tokens) * cache_tokens_per_history_step
    )
    cfg = OnlineHistoryRuntimeConfig(
        patch_size=patch_size,
        patch_stride=patch_stride,
        context_half=context_half,
        decode_patch_size=decode_patch_size,
        attention_history_tokens=attention_history_tokens,
        attention_history_cache_tokens=attention_history_cache_tokens,
        cache_tokens_per_history_step=cache_tokens_per_history_step,
        use_rope=use_rope,
        rope_base=rope_base,
        arm_dof=int(arm_dof),
        emb_dim=int(emb_dim),
        jointwise=True,
    )
    cache0 = models_transformer.init_infer_state(
        params_hist_example,
        hist_model,
        batch_size=1,
    )

    @jax.jit
    def decode_step(
        params_hist,
        cache,
        q_patch,
        qd_patch,
        tau_patch,
        input_keep_mask,
        norm_stats,
    ):
        valid_mask = jnp.ones(q_patch.shape[:2], dtype=jnp.float32)
        cache = limit_decode_cache_attention_window(
            cache,
            attention_history_cache_tokens,
            append_tokens=cache_tokens_per_history_step,
            rebase_rope=use_rope,
            rope_base=rope_base,
        )
        emb, cache_out = models_transformer.step_decode(
            params=params_hist,
            cache=cache,
            model=hist_model,
            chunk_q=q_patch,
            chunk_qd=qd_patch,
            chunk_u=tau_patch,
            valid_mask=valid_mask,
            key=None,
            norm_stats=norm_stats,
            input_keep_mask=input_keep_mask,
        )
        return emb[:, 0, ...], cache_out

    return OnlineHistoryRuntime(config=cfg, cache0=cache0, decode_step=decode_step)


def init_online_history_state(
    runtime: OnlineHistoryRuntime,
    *,
    dtype=jnp.float32,
) -> OnlineHistoryState:
    cfg = runtime.config
    dummy_emb = (
        jnp.zeros((1, cfg.arm_dof, cfg.emb_dim), dtype=dtype)
        if cfg.jointwise
        else jnp.zeros((1, cfg.emb_dim), dtype=dtype)
    )
    q0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    qd0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    tau0 = jnp.zeros((cfg.decode_patch_size, cfg.arm_dof), dtype=dtype)
    keep0 = jnp.zeros((cfg.decode_patch_size,), dtype=dtype)
    return OnlineHistoryState(
        history_emb=dummy_emb,
        cache=runtime.cache0,
        q_buf=q0,
        qd_buf=qd0,
        tau_buf=tau0,
        keep_buf=keep0,
        sample_count=jnp.asarray(1, dtype=jnp.int32),
        next_emit_idx=jnp.asarray(cfg.patch_size - 1, dtype=jnp.int32),
        has_embedding=jnp.asarray(False),
    )


def advance_online_history_state(
    runtime: OnlineHistoryRuntime,
    state: OnlineHistoryState,
    *,
    q_arm: jax.Array,
    qd_arm: jax.Array,
    tau_arm: jax.Array,
    raw_tau_arm: jax.Array,
    params_hist,
    norm_stats,
) -> OnlineHistoryState:
    q_masked, qd_masked, tau_masked, keep = mask_zero_torque_history_sample(
        q_arm,
        qd_arm,
        tau_arm,
        raw_tau_arm,
    )
    pushed = state.replace(
        q_buf=push_window(state.q_buf, q_masked),
        qd_buf=push_window(state.qd_buf, qd_masked),
        tau_buf=push_window(state.tau_buf, tau_masked),
        keep_buf=push_window(state.keep_buf, keep),
        sample_count=state.sample_count + 1,
    )
    should_emit = (pushed.sample_count - 1) >= (
        pushed.next_emit_idx + runtime.config.context_half
    )

    def emit_token(st: OnlineHistoryState) -> OnlineHistoryState:
        q_patch = st.q_buf[None, None, ...]
        qd_patch = st.qd_buf[None, None, ...]
        tau_patch = st.tau_buf[None, None, ...]
        keep_patch = st.keep_buf[None, None, ...]
        history_emb, cache = runtime.decode_step(
            params_hist,
            st.cache,
            q_patch,
            qd_patch,
            tau_patch,
            keep_patch,
            align_norm_stats_to_dof(norm_stats, runtime.config.arm_dof),
        )
        return st.replace(
            history_emb=history_emb,
            cache=cache,
            next_emit_idx=st.next_emit_idx + runtime.config.patch_stride,
            has_embedding=jnp.asarray(True),
        )

    return jax.lax.cond(should_emit, emit_token, lambda st: st, pushed)


def apply_online_adaptor(
    *,
    adaptor_model,
    adaptor_apply_fn: Callable[..., tuple[jax.Array, Any]] | None = None,
    params_adaptor,
    q_window: jax.Array,
    qd_window: jax.Array,
    tau_window: jax.Array,
    history_emb: jax.Array,
    norm_stats,
) -> tuple[jax.Array, jax.Array]:
    norm_stats = align_norm_stats_to_dof(norm_stats, int(tau_window.shape[-1]))
    if adaptor_apply_fn is None:
        delta_tau, _ = adaptor_model.apply(
            params_adaptor,
            q_window[None, ...],
            qd_window[None, ...],
            tau_window[None, ...],
            history_emb,
            norm_stats=norm_stats,
        )
    else:
        delta_tau, _ = adaptor_apply_fn(
            params_adaptor,
            q_window[None, ...],
            qd_window[None, ...],
            tau_window[None, ...],
            history_emb,
            jax.random.PRNGKey(0),
            False,
            norm_stats,
        )
    delta_tau = delta_tau[0]
    return tau_window[-1] + delta_tau, delta_tau
