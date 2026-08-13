from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import einops
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.attention import dot_product_attention

from simadaptor.models.adaptor import HistoryEmbed
import simadaptor.physics.dynamics as dynamics

from mujoco import mjx
from simadaptor.config import EncoderConfig

# -----------------------------
# Mask utilities
# -----------------------------
def make_causal_mask(L: int) -> jnp.ndarray:
    """[1,1,L,L] causal mask (1=allow, 0=block)."""
    i = jnp.arange(L)[:, None]
    j = jnp.arange(L)[None, :]
    m = (j <= i).astype(jnp.float32)     # lower triangle incl. diagonal
    return m[None, None, :, :]

def make_local_causal_mask(L: int, window: int) -> jnp.ndarray:
    """[1,1,L,L] local causal mask with lookback < window."""
    i = jnp.arange(L)[:, None]
    j = jnp.arange(L)[None, :]
    m = ((j <= i) & ((i - j) < window)).astype(jnp.float32)
    return m[None, None, :, :]

def make_joint_time_causal_mask(n_steps: int, dof: int) -> jnp.ndarray:
    """
    [1,1,L,L] causal mask for flattened joint tokens (L = n_steps*dof),
    where causality is enforced on time index only (token_time = idx // dof).
    This allows all joints at the same time step to attend each other.
    """
    L = n_steps * dof
    i = jnp.arange(L)[:, None]
    j = jnp.arange(L)[None, :]
    ti = i // dof
    tj = j // dof
    m = (tj <= ti).astype(jnp.float32)
    return m[None, None, :, :]

def make_joint_time_local_causal_mask(n_steps: int, dof: int, window_steps: int) -> jnp.ndarray:
    """
    [1,1,L,L] local causal mask for flattened joint tokens with a time-window lookback.
    `window_steps` is in units of history-token time steps (not flattened token count).
    """
    L = n_steps * dof
    i = jnp.arange(L)[:, None]
    j = jnp.arange(L)[None, :]
    ti = i // dof
    tj = j // dof
    m = ((tj <= ti) & ((ti - tj) < window_steps)).astype(jnp.float32)
    return m[None, None, :, :]

def make_padding_mask(valid_1d: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
    """
    valid_1d: [B, L] with 1 for valid, 0 for pad.
    Returns [B,1,1,L] or None.
    """
    if valid_1d is None:
        return None
    return valid_1d[:, None, None, :].astype(jnp.float32)


def make_decode_query_mask(valid_1d: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
    """
    valid_1d: [B, Lq] with 1 for valid current-query tokens, 0 for invalid.
    Returns [B,1,Lq,1] or None so it broadcasts against KV-cache key masks [1,1,1,Sk].
    """
    if valid_1d is None:
        return None
    return valid_1d[:, None, :, None].astype(jnp.float32)

def combine_masks(*masks: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    out = masks[0]
    for m in masks[1:]:
        out = out * m
    return out


# ---- Rotary positional embeddings (RoPE) helpers ----

def rope_angles(head_dim: int, base: float = 10000.0) -> jnp.ndarray:
    half = head_dim // 2
    idx = jnp.arange(half, dtype=jnp.float32)
    return 1.0 / (base ** (idx / half))

def rope_cos_sin(max_len: int, head_dim: int, base: float = 10000.0):
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) / half))  # [Hd/2]
    pos = jnp.arange(max_len, dtype=jnp.float32)[:, None]               # [Lmax,1]
    ang = pos * inv[None, :]                                            # [Lmax,Hd/2]
    return jnp.cos(ang), jnp.sin(ang)                                   # each [Lmax,Hd/2]

def apply_rope_qk(x, cos_half, sin_half, positions):
    # x: [B,L,H,Hd], cos/sin_half: [Lmax,Hd/2], positions: [L]
    cos_t = cos_half[positions][None, :, None, :]  # [1,L,1,Hd/2]
    sin_t = sin_half[positions][None, :, None, :]  # [1,L,1,Hd/2]

    x_even = x[..., ::2]            # [B,L,H,Hd/2]
    x_odd  = x[..., 1::2]           # [B,L,H,Hd/2]

    xr_even = x_even * cos_t - x_odd * sin_t
    xr_odd  = x_even * sin_t + x_odd * cos_t

    return jnp.reshape(
        jnp.stack([xr_even, xr_odd], axis=-1),   # [B,L,H,Hd/2,2]
        x.shape,                                 # [B,L,H,Hd]
    )


def apply_rope_qk_at_positions(x, positions, base: float = 10000.0):
    """Apply RoPE at arbitrary (including decode-stream absolute) positions."""
    head_dim = int(x.shape[-1])
    if head_dim % 2:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}.")
    half = head_dim // 2
    inv = 1.0 / (
        float(base) ** (jnp.arange(half, dtype=jnp.float32) / max(half, 1))
    )
    angle = jnp.asarray(positions, dtype=jnp.float32)[:, None] * inv[None, :]
    cos_t = jnp.cos(angle).astype(x.dtype)[None, :, None, :]
    sin_t = jnp.sin(angle).astype(x.dtype)[None, :, None, :]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos_t - x_odd * sin_t
    rotated_odd = x_even * sin_t + x_odd * cos_t
    return jnp.reshape(
        jnp.stack([rotated_even, rotated_odd], axis=-1),
        x.shape,
    )


class RoPESelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0
    # RoPE
    rope_base: float = 10000.0
    rope_max_len: int = 128

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,                      # [B, T, D]
                 attention_mask: jnp.ndarray | None,  # broadcastable to [B, H, T, T]
                 *,
                 deterministic: bool,
                 decode: bool) -> jnp.ndarray:
        B, T, D = x.shape
        H = self.num_heads
        Hd = D // H
        assert D % H == 0, "d_model must be divisible by num_heads"

        # Project QKV (stay as [B,T,D] then reshape to [B,T,H,Hd])
        qkv = nn.Dense(3 * D, use_bias=True, name="qkv")(x)              # [B,T,3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)                             # each [B,T,D]
        def to_bthh(t): return t.reshape(B, T, H, Hd)                    # [B,T,H,Hd]
        q = to_bthh(q); k = to_bthh(k); v = to_bthh(v)

        # RoPE tables (once)
        cos_tbl, sin_tbl = rope_cos_sin(self.rope_max_len, Hd, self.rope_base)
        cos_tbl = cos_tbl.astype(x.dtype)
        sin_tbl = sin_tbl.astype(x.dtype)
        attn_dropout_rng = None if (deterministic or self.dropout_rate == 0.0) else self.make_rng("dropout")

        if decode:
            # --- KV cache: [B, Smax, H, Hd] ---
            k_cache = self.variable("cache", "cached_key",
                                    lambda: jnp.zeros((B, self.rope_max_len, H, Hd), x.dtype))
            v_cache = self.variable("cache", "cached_value",
                                    lambda: jnp.zeros((B, self.rope_max_len, H, Hd), x.dtype))
            idx_var = self.variable("cache", "cache_index",
                                    lambda: jnp.array(0, dtype=jnp.int32))
            rope_idx_var = self.variable(
                "cache",
                "cache_rope_index",
                lambda: jnp.array(0, dtype=jnp.int32),
            )

            start = idx_var.value
            rope_start = rope_idx_var.value
            # Clamp write position to stay within cache; avoid Python conditionals on traced values.
            start_write = jnp.clip(start, 0, jnp.maximum(0, self.rope_max_len - T))
            rope_pos = jnp.arange(T, dtype=jnp.int32) + rope_start
            q = apply_rope_qk_at_positions(q, rope_pos, self.rope_base)  # [B,T,H,Hd]
            k = apply_rope_qk_at_positions(k, rope_pos, self.rope_base)  # [B,T,H,Hd]

            # write to cache at [start:start+T]
            indices = jnp.arange(T) + start_write
            k_cache.value = k_cache.value.at[:, indices, :, :].set(k)     # [B,Smax,H,Hd]
            v_cache.value = v_cache.value.at[:, indices, :, :].set(v)
            # advance index but keep it bounded by rope_max_len to prevent overflow
            idx_var.value = jnp.minimum(start + T, self.rope_max_len)
            # Keep an absolute RoPE position independent of physical cache shifts.
            rope_idx_var.value = rope_start + T

            # Use full cache but mask out entries beyond current index to avoid dynamic slice sizes.
            K = k_cache.value  # [B,Smax,H,Hd]
            V = v_cache.value
            valid_len = idx_var.value
            key_mask = (jnp.arange(self.rope_max_len, dtype=jnp.int32) < valid_len)[None, None, None, :]  # [1,1,1,Smax]
            key_mask = key_mask.astype(x.dtype)
            attn_mask = key_mask if attention_mask is None else attention_mask * key_mask
            # dot_product_attention expects:
            #   Q: [B, T, H, Hd], K/V: [B, S, H, Hd], mask: [B, H, T, S]
            attn_out = dot_product_attention(
                query=q, key=K, value=V,
                bias=None,
                mask=attn_mask,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=x.dtype,
                dropout_rng=attn_dropout_rng,
            )  # [B,T,H,Hd]
        else:
            # training: positions 0..T-1 for this chunk
            pos = jnp.arange(T, dtype=jnp.int32)
            q = apply_rope_qk(q, cos_tbl, sin_tbl, pos)
            k = apply_rope_qk(k, cos_tbl, sin_tbl, pos)
            attn_out = dot_product_attention(
                query=q, key=k, value=v,
                bias=None,
                mask=attention_mask,       # causal × (optional local) × padding
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=x.dtype,
                dropout_rng=attn_dropout_rng,
            )

        # merge heads -> [B,T,D]
        y = attn_out.reshape(B, T, D)
        y = nn.Dense(D, use_bias=True, name="out")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        return y


class MLP(nn.Module):
    d_model: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        hidden = int(self.d_model * self.mlp_ratio)
        y = nn.Dense(hidden, name="fc1")(x)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.d_model, name="fc2")(y)
        y = nn.Dropout(self.dropout)(y, deterministic=deterministic)
        return y


class DecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_base: float = 10000.0
    rope_max_len: int = 128

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        *,
        deterministic: bool,
        decode: bool,
    ) -> jnp.ndarray:
        # Pre-LN + self-attn
        y = nn.LayerNorm(name="ln1")(x)
        y = RoPESelfAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout,
            rope_base=self.rope_base,
            rope_max_len=self.rope_max_len,
        )(y, attention_mask=attention_mask, deterministic=deterministic, decode=decode)
        x = x + nn.Dropout(self.dropout)(y, deterministic=deterministic)

        # MLP
        y = nn.LayerNorm(name="ln2")(x)
        y = MLP(self.d_model, self.mlp_ratio, self.dropout)(y, deterministic=deterministic)
        x = x + y
        return x


class ARTransformerDecoder(nn.Module):
    """Autoregressive history encoder for global SimAdaptor embeddings.

    This module wraps `HistoryEmbed` and a causal Transformer stack.

    Input shapes:
    - Offline / training path: `q`, `qd`, `u` are `[B, T, DoF]`
    - Online decode path: `q`, `qd`, `u` are `[B, 1, P, DoF]`

    Torque semantics:
    - `u` is the logged torque history that was actually applied to the robot,
      i.e. `tau_cmd`.
    - It is not just the raw pre-adaptor desired torque.

    Output:
    - Global history tokens `[B, N, C]`, where `N` is the number of temporal
      patches after `HistoryEmbed` chunking and `C = emb_dim`.
    """
    cfg: EncoderConfig
    emb_dim: int
    ideal_mjx_model: Optional[mjx.MjModel] = None

    def setup(self):
        if self.ideal_mjx_model is None:
            raise ValueError("ARTransformerDecoder requires ideal_mjx_model.")
        # implement my own embeddings
        self.embed = HistoryEmbed(emb_dim=self.cfg.d_model, 
                                  patch_size=self.cfg.patch_size, 
                                  patch_stride=self.cfg.patch_stride, 
                                  jointwise=False,
                                  masked_fit_max_neighbors_each_side=self.cfg.masked_fit_max_neighbors_each_side,
                                  masked_fit_q_weight=self.cfg.masked_fit_q_weight,
                                  masked_fit_qd_weight=self.cfg.masked_fit_qd_weight,
                                  ideal_mjx_model=self.ideal_mjx_model)
        
        self.blocks = [
            DecoderBlock(
                d_model=self.cfg.d_model,
                num_heads=self.cfg.num_heads,
                mlp_ratio=self.cfg.mlp_ratio,
                dropout=self.cfg.dropout,
                rope_base=self.cfg.rope_base,
                rope_max_len=self.cfg.rope_max_len,
            ) for _ in range(self.cfg.num_layers)
        ]

    @nn.compact
    def transformer_only(
        self,
        x: jnp.ndarray,
        attn_valid_1d: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = False,
        decode: bool = False,
        train_window_override: Optional[int] = None,
    ) -> jnp.ndarray:
        """Run only the transformer stack on precomputed history-embed tokens."""
        B, L = x.shape[:-1]
        del B
        x = nn.Dropout(self.cfg.emb_dropout)(x, deterministic=deterministic)

        if decode:
            attn_mask = make_decode_query_mask(attn_valid_1d)
        else:
            causal = make_causal_mask(L)
            win = train_window_override if train_window_override is not None else self.cfg.train_window
            if win is not None:
                causal = causal * make_local_causal_mask(L, win)
            pad = make_padding_mask(attn_valid_1d)
            attn_mask = combine_masks(causal, pad)

        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask, deterministic=deterministic, decode=decode)

        x = nn.LayerNorm(name="final_ln")(x)
        x = nn.Dense(self.emb_dim, name="final_proj")(x)
        return x

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        qd: jnp.ndarray,
        u: jnp.ndarray,
        attn_valid_1d: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = False,
        decode: bool = False,
        train_window_override: Optional[int] = None,
        norm_stats: Optional[dict] = None,
        input_keep_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Encode history into global latent tokens.

        Args:
            q, qd, u:
                Either full histories `[B, T, DoF]` or a single decode patch
                `[B, 1, P, DoF]`. `u` is the applied torque history `tau_cmd`.
            attn_valid_1d:
                Optional token-valid mask `[B, N]` after history chunking. In
                decode mode this is typically `[B, 1]`.
        Returns:
            Token embeddings with shape `[B, N, emb_dim]`.
        """
        x = self.embed(
            q,
            qd,
            u,
            deterministic=deterministic,
            norm_stats=norm_stats,
            input_keep_mask=input_keep_mask,
        )          # [B,L,D]
        return self.transformer_only(
            x,
            attn_valid_1d,
            deterministic=deterministic,
            decode=decode,
            train_window_override=train_window_override,
        )


class JointwiseFlatARTransformerDecoder(nn.Module):
    """Autoregressive history encoder for jointwise SimAdaptor embeddings.

    Input semantics match `ARTransformerDecoder`, but `HistoryEmbed` first
    produces jointwise tokens `[B, N, DoF, C]` and this module flattens the
    `(N, DoF)` token grid into one causal stream before Transformer decoding.
    """
    cfg: EncoderConfig
    emb_dim: int
    ideal_mjx_model: Optional[mjx.MjModel] = None

    def setup(self):
        if self.ideal_mjx_model is None:
            raise ValueError("JointwiseFlatARTransformerDecoder requires ideal_mjx_model.")
        self.embed = HistoryEmbed(
            emb_dim=self.cfg.d_model,
            patch_size=self.cfg.patch_size,
            patch_stride=self.cfg.patch_stride,
            jointwise=True,
            masked_fit_max_neighbors_each_side=self.cfg.masked_fit_max_neighbors_each_side,
            masked_fit_q_weight=self.cfg.masked_fit_q_weight,
            masked_fit_qd_weight=self.cfg.masked_fit_qd_weight,
            ideal_mjx_model=self.ideal_mjx_model,
        )
        self.blocks = [
            DecoderBlock(
                d_model=self.cfg.d_model,
                num_heads=self.cfg.num_heads,
                mlp_ratio=self.cfg.mlp_ratio,
                dropout=self.cfg.dropout,
                rope_base=self.cfg.rope_base,
                rope_max_len=self.cfg.rope_max_len,
            )
            for _ in range(self.cfg.num_layers)
        ]

    @nn.compact
    def transformer_only(
        self,
        x: jnp.ndarray,
        attn_valid_1d: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = False,
        decode: bool = False,
        train_window_override: Optional[int] = None,
    ) -> jnp.ndarray:
        """Run only the transformer stack on precomputed jointwise history tokens."""
        if x.ndim != 4:
            raise ValueError(
                f"JointwiseFlatARTransformerDecoder expects rank-4 tokens [B,N,DoF,C], got {x.shape}"
            )
        B, N, dof, _ = x.shape

        max_dof_tokens = int(getattr(self.cfg, "max_dof_tokens", 0) or 0)
        if max_dof_tokens < 1:
            raise ValueError(f"cfg.max_dof_tokens must be >= 1, got {max_dof_tokens}")
        if dof > max_dof_tokens:
            raise ValueError(
                f"Input DoF={dof} exceeds cfg.max_dof_tokens={max_dof_tokens}. "
                "Increase enc.max_dof_tokens for mixed-embodiment training."
            )

        joint_id_table = self.param(
            "joint_id_embedding",
            nn.initializers.normal(stddev=0.02),
            (max_dof_tokens, self.cfg.d_model),
        )
        joint_id = joint_id_table[:dof]
        x = x + joint_id[None, None, :, :]

        x = einops.rearrange(x, "b n d c -> b (n d) c")
        L = x.shape[1]
        x = nn.Dropout(self.cfg.emb_dropout)(x, deterministic=deterministic)

        attn_valid_flat = None
        if attn_valid_1d is not None:
            if attn_valid_1d.ndim != 2:
                raise ValueError(f"attn_valid_1d must be [B,L] or [B,N], got {attn_valid_1d.shape}")
            if attn_valid_1d.shape[1] == N:
                attn_valid_flat = jnp.repeat(attn_valid_1d, dof, axis=1)
            elif attn_valid_1d.shape[1] == L:
                attn_valid_flat = attn_valid_1d
            else:
                raise ValueError(
                    f"attn_valid_1d length must be N={N} or N*DoF={L}, got {attn_valid_1d.shape[1]}"
                )

        if decode:
            attn_mask = make_decode_query_mask(attn_valid_flat)
        else:
            causal = make_joint_time_causal_mask(N, dof)
            win = train_window_override if train_window_override is not None else self.cfg.train_window
            if win is not None:
                win = int(win)
                if win < 1:
                    raise ValueError(f"train_window must be >= 1 when set; got {win}")
                causal = causal * make_joint_time_local_causal_mask(N, dof, win)
            pad = make_padding_mask(attn_valid_flat)
            attn_mask = combine_masks(causal, pad)

        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask, deterministic=deterministic, decode=decode)

        x = nn.LayerNorm(name="final_ln")(x)
        x = nn.Dense(self.emb_dim, name="final_proj")(x)
        x = einops.rearrange(x, "b (n d) e -> b n d e", n=N, d=dof)
        return x

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        qd: jnp.ndarray,
        u: jnp.ndarray,
        attn_valid_1d: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = False,
        decode: bool = False,
        train_window_override: Optional[int] = None,
        norm_stats: Optional[dict] = None,
        input_keep_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        x = self.embed(
            q,
            qd,
            u,
            deterministic=deterministic,
            norm_stats=norm_stats,
            input_keep_mask=input_keep_mask,
        )
        return self.transformer_only(
            x,
            attn_valid_1d,
            deterministic=deterministic,
            decode=decode,
            train_window_override=train_window_override,
        )


def init_infer_state(
    params: Dict[str, Any],
    model: ARTransformerDecoder,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Initialize mutable cache by calling the model once in decode mode with a fake token.

    Flax creates cache variables lazily on the first decode call. We use a fake
    token only to materialize the cache shapes, then explicitly clear the
    written key/value entries so the returned cache represents an empty decode
    history rather than a one-token-prefilled state.
    """
    ideal_mjx_model = getattr(model, "ideal_mjx_model", None)
    dof = int(getattr(ideal_mjx_model, "nu", 0) or getattr(ideal_mjx_model, "nv", 0) or getattr(ideal_mjx_model, "nq", 0))
    if dof <= 0:
        raise ValueError(
            "Could not infer model DoF for decode cache initialization. "
            "Expected model.ideal_mjx_model to expose nu/nv/nq."
        )

    fake_q = jnp.zeros((batch_size, 1, model.cfg.patch_size, dof), dtype=jnp.float32)
    fake_qd = jnp.zeros((batch_size, 1, model.cfg.patch_size, dof), dtype=jnp.float32)
    fake_u = jnp.zeros((batch_size, 1, model.cfg.patch_size, dof), dtype=jnp.float32)
    fake_mask = jnp.ones((batch_size, 1), dtype=jnp.float32)

    _, vars_out = model.apply(
        params,
        fake_q, fake_qd, fake_u,
        fake_mask,
        deterministic=True,
        decode=True,
        mutable=["cache"],
        rngs={"dropout": jax.random.PRNGKey(0)},
    )

    def _clear_cache_tree(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key in (
                    "cached_key",
                    "cached_value",
                    "cache_index",
                    "cache_rope_index",
                ):
                    out[key] = jnp.zeros_like(value)
                else:
                    out[key] = _clear_cache_tree(value)
            return out
        return node

    return {"cache": _clear_cache_tree(vars_out.get("cache", {}))}


def step_decode(
    params: Dict[str, Any],
    cache: Dict[str, Any],
    model: ARTransformerDecoder,
    chunk_q, chunk_qd, chunk_u,
    valid_mask: Optional[jnp.ndarray] = None,  # [B,1] or None
    key: Optional[jax.random.PRNGKey] = None,
    norm_stats: Optional[dict] = None,
    input_keep_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    One decoding step: feeds a single token and returns its embedding plus updated cache.
    Output is [B,1,E] for global models or [B,1,DoF,E] for jointwise models.
    """
    # if valid_mask is None:
    #     valid_mask = jnp.ones_like(next_token_id, dtype=jnp.float32)

    variables = {"params": params['params'], **cache}
    hidden, new_vars = model.apply(
        variables,
        chunk_q, chunk_qd, chunk_u,
        valid_mask,
        deterministic=True,
        decode=True,
        norm_stats=norm_stats,
        input_keep_mask=input_keep_mask,
        mutable=["cache"],
        rngs={"dropout": jax.random.PRNGKey(0) if key is None else key},
    )
    last_hidden = hidden[:, -1:, :]   # [B,1,D]
    return last_hidden, new_vars


def online_history_update(
    params: Dict[str, Any],
    model: ARTransformerDecoder,
    q_seq,
    qd_seq,
    u_seq,
    cache: Optional[Dict[str, Any]] = None,
    valid_mask: Optional[jnp.ndarray] = None,
    key: Optional[jax.random.PRNGKey] = None,
    norm_stats: Optional[dict] = None,
    input_keep_mask: Optional[jnp.ndarray] = None,
):
    """
    Autoregressively update history embeddings using the decoder in ``decode`` mode.

    Args:
        params: Parameter PyTree for ``model`` (expects a ``"params"`` key).
        model: History embedding transformer.
        q_seq/qd_seq/u_seq: Token inputs shaped [B, T, P, D] or [T, P, D]; P should
            match ``cfg.patch_size``. A missing batch dim (or both batch/time) will
            be added automatically, e.g., passing [P, D] is treated as a single
            token with batch=1, and [T, P, D] is treated as batch=1.
        cache: Mutable attention cache returned by ``init_infer_state``. If None, a
            fresh cache is initialized for the given batch size.
        valid_mask: Optional validity mask shaped [B, T] (or broadcastable). Per-step
            masks are passed to attention during decoding.
        key: Optional PRNGKey; when provided it is split per step for dropout rngs.

    Returns:
        history_emb: [B, T, emb_dim] embeddings for each decoded token.
        cache: Updated decode cache to be fed into the next call.
    """

    def _normalize(x):
        # Ensure input is [B, T, P, D] following the docstring contract.
        if x.ndim == 2:            # [P, D]
            x = x[None, None, ...]
        elif x.ndim == 3:          # [T, P, D]
            x = x[None, ...]
        elif x.ndim != 4:
            raise ValueError(f"Expected 2-4 dims for token inputs, got shape {x.shape}")
        return x

    q_tokens = _normalize(q_seq)
    qd_tokens = _normalize(qd_seq)
    u_tokens = _normalize(u_seq)

    if q_tokens.shape != qd_tokens.shape or q_tokens.shape != u_tokens.shape:
        raise ValueError("q_seq, qd_seq, and u_seq must share shape after normalization.")

    B, T, P, _ = q_tokens.shape
    if P != model.cfg.patch_size:
        raise ValueError(f"Patch length {P} does not match model patch_size {model.cfg.patch_size}.")

    if cache is None:
        cache = init_infer_state(params, model, batch_size=B)

    if valid_mask is None:
        masks = jnp.ones((B, T, 1), dtype=jnp.float32)
    else:
        masks = valid_mask
        if masks.ndim == 1:
            masks = masks[None, :, None]
        elif masks.ndim == 2:
            masks = masks[..., None]
        if masks.shape[0] != B or masks.shape[1] != T:
            raise ValueError(f"valid_mask must broadcast to [B, T]; got {masks.shape} vs {(B, T)}")
        masks = masks.astype(jnp.float32)

    # time-major for scan: [T, B, ...]
    q_tokens_t = jnp.swapaxes(q_tokens, 0, 1)
    qd_tokens_t = jnp.swapaxes(qd_tokens, 0, 1)
    u_tokens_t = jnp.swapaxes(u_tokens, 0, 1)
    masks_t = jnp.swapaxes(masks, 0, 1)

    keep_tokens_t = None
    if input_keep_mask is not None:
        keep_tokens = input_keep_mask
        if keep_tokens.ndim == 2:
            keep_tokens = keep_tokens[None, ..., None]
        elif keep_tokens.ndim == 3:
            if keep_tokens.shape[:2] == (B, T):
                keep_tokens = keep_tokens[..., None]
            elif keep_tokens.shape[0] != B or keep_tokens.shape[1] != T:
                raise ValueError(
                    f"input_keep_mask rank-3 must match [B,T,P] or [B,T,1], got {keep_tokens.shape} vs {(B, T)}"
                )
        elif keep_tokens.ndim != 4:
            raise ValueError(
                f"input_keep_mask must have rank 2-4 and broadcast to [B,T,P,1], got {keep_tokens.shape}"
            )
        if keep_tokens.shape[0] != B or keep_tokens.shape[1] != T:
            raise ValueError(
                f"input_keep_mask must match [B,T,...], got {keep_tokens.shape} vs {(B, T)}"
            )
        if keep_tokens.shape[-2] != P:
            if keep_tokens.shape[-2] == 1:
                keep_tokens = jnp.broadcast_to(keep_tokens, (B, T, P, keep_tokens.shape[-1]))
            else:
                raise ValueError(
                    f"input_keep_mask patch axis must match P={P}, got {keep_tokens.shape}"
                )
        if keep_tokens.shape[-1] == 1:
            keep_tokens = keep_tokens[..., 0]
        elif keep_tokens.shape[-1] != 1:
            raise ValueError(
                f"input_keep_mask trailing axis must be 1 when rank-4, got {keep_tokens.shape}"
            )
        keep_tokens = keep_tokens.astype(jnp.float32)
        keep_tokens_t = jnp.swapaxes(keep_tokens, 0, 1)

    if key is not None:
        keys = jax.random.split(key, T)

        def step_fn(carry, inputs):
            q_t, qd_t, u_t, m_t, keep_t, k_t = inputs
            h_t, new_cache = step_decode(
                params,
                carry,
                model,
                q_t,
                qd_t,
                u_t,
                valid_mask=m_t,
                key=k_t,
                norm_stats=norm_stats,
                input_keep_mask=keep_t,
            )
            return new_cache, h_t[:, 0, :]  # squeeze the length-1 token axis

        if keep_tokens_t is None:
            keep_tokens_t = jnp.ones((T, B, P), dtype=jnp.float32)
        cache, h_stack = jax.lax.scan(step_fn, cache, (q_tokens_t, qd_tokens_t, u_tokens_t, masks_t, keep_tokens_t, keys))
    else:
        def step_fn(carry, inputs):
            q_t, qd_t, u_t, m_t, keep_t = inputs
            h_t, new_cache = step_decode(
                params,
                carry,
                model,
                q_t,
                qd_t,
                u_t,
                valid_mask=m_t,
                key=None,
                norm_stats=norm_stats,
                input_keep_mask=keep_t,
            )
            return new_cache, h_t[:, 0, :]

        if keep_tokens_t is None:
            keep_tokens_t = jnp.ones((T, B, P), dtype=jnp.float32)
        cache, h_stack = jax.lax.scan(step_fn, cache, (q_tokens_t, qd_tokens_t, u_tokens_t, masks_t, keep_tokens_t))

    history_emb = jnp.swapaxes(h_stack, 0, 1)  # [B, T, emb_dim]
    return history_emb, cache


def trim_cache_window(cache: Dict[str, Any], keep: int) -> Dict[str, Any]:
    """
    Optional: keep only the most recent `keep` KV entries in SelfAttention caches.
    This prevents unbounded memory/time as sequences grow.
    Works by slicing cached_key/value along the sequence axis and fixing cache_index.
    """
    cache_out = {"cache": {}}
    for mod_name, mod_vars in cache["cache"].items():
        mod_out = {}
        for k, v in mod_vars.items():
            # Typical keys: "cached_key", "cached_value", "cache_index"
            if k in ("cached_key", "cached_value"):
                # shape usually [B, num_heads, length, head_dim]
                length_axis = 2
                cur_len = v.shape[length_axis]
                if cur_len > keep:
                    v = jnp.take(v, indices=jnp.arange(cur_len - keep, cur_len), axis=length_axis)
                mod_out[k] = v
            elif k == "cache_index":
                # reset index to the (possibly trimmed) length
                # (Flax increments this internally each decode step)
                if "cached_key" in mod_vars:
                    new_len = mod_out.get("cached_key", mod_vars["cached_key"]).shape[2]
                else:
                    new_len = mod_vars[k].shape[0]  # fallback
                mod_out[k] = jnp.array(new_len, dtype=mod_vars[k].dtype)
            else:
                mod_out[k] = v
        cache_out["cache"][mod_name] = mod_out
    return cache_out


if __name__ == "__main__":
    # Build model & params
    cfg = EncoderConfig(
        d_model=512,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1,
        train_window=None,   # or an int like 1024
    )
    model = ARTransformerDecoder(cfg)

    key = jax.random.PRNGKey(0)
    B, L = 4, 1000
    # token_ids = jnp.zeros((B, L), dtype=jnp.int32)
    q = jax.random.normal(key, (B, L, 7))
    qd = jax.random.normal(key, (B, L, 7))
    u = jax.random.normal(key, (B, L, 7))
    
    valid = None
    params = model.init({"params": key, "dropout": key}, q, qd, u, valid, deterministic=False, decode=False)
    
    # inference loop
    # params, model already created
    cache = init_infer_state(params, model, max_len=cfg.max_len, batch_size=B)

    hidden_steps = []
    for t in range(10):
        q_t = jax.random.normal(key, (B, model.cfg.patch_size, 7))
        qd_t = jax.random.normal(key, (B, model.cfg.patch_size, 7))
        u_t = jax.random.normal(key, (B, model.cfg.patch_size, 7))
        h_t, cache = step_decode(params, cache, model, q_t, qd_t, u_t)  # h_t: [B,1,D]
        hidden_steps.append(h_t)

        # Optional: sliding window for long streams
        # cache = trim_cache_window(cache, keep=512)

    hidden_seq = jnp.concatenate(hidden_steps, axis=1)  # [B,T,D]

    # Forward train
    rng = jax.random.PRNGKey(1)
    hidden, _ = forward_train(params, model, q, qd, u, valid, rng)  # [B,L,D]
    
    print(1)
    
    
    
    
    
