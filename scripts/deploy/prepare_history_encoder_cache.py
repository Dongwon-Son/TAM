#!/usr/bin/env python3
"""Pre-warm the deploy history encoder JAX compilation cache."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np

from simadaptor.deploy.history_runtime import (
    DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
    RealTimeHistoryAdaptor,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a SimAdaptor checkpoint and compile the deploy history-decoder "
            "shape into JAX's persistent cache before starting mapping_server."
        )
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        required=True,
        help="Local checkpoint_<step> directory to load.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("assets/franka_panda/panda_pandagripper.xml"),
        help="MuJoCo XML used to resolve the robot model, matching mapping_server default.",
    )
    parser.add_argument("--expected-dt", type=float, default=0.001)
    parser.add_argument(
        "--attention-history-s",
        type=float,
        default=DEFAULT_DEPLOY_ATTENTION_HISTORY_S,
        help=(
            "Match mapping_server --attention-history-s "
            f"(default: {DEFAULT_DEPLOY_ATTENTION_HISTORY_S:g})."
        ),
    )
    parser.add_argument(
        "--history-jax-cache-dir",
        type=Path,
        default=Path(".cache") / "jax_history_compile",
        help="Persistent JAX compilation cache directory to populate.",
    )
    parser.add_argument(
        "--history-jax-cache-min-compile-time-s",
        type=float,
        default=0.0,
        help="Compile-time threshold for cache writes; keep 0 for deploy prewarm.",
    )
    parser.add_argument(
        "--history-jax-cache-min-entry-size-bytes",
        type=int,
        default=-1,
        help="Entry-size threshold for cache writes; keep -1 for deploy prewarm.",
    )
    parser.add_argument(
        "--history-jax-cache-xla-caches",
        type=str,
        default="xla_gpu_per_fusion_autotune_cache_dir",
        help="Extra XLA caches to persist; set empty to leave unset.",
    )
    parser.add_argument(
        "--smoke-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one synthetic streaming window after constructor warm-up.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    start = time.perf_counter()
    adaptor = RealTimeHistoryAdaptor(
        simadaptor_ckpt_path=str(args.ckpt_path),
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
    if bool(args.smoke_window) and not bool(getattr(adaptor, "_constant_zero_history", False)):
        n = int(getattr(adaptor, "_decode_patch_size")) + int(getattr(adaptor, "_patch_stride"))
        t = np.arange(n, dtype=np.float64) * float(args.expected_dt)
        q = np.zeros((n, adaptor._dof), dtype=np.float32)
        qd = np.zeros_like(q)
        tau = np.full_like(q, 1.0e-3)
        emb = adaptor.push_window(
            timestamps=t,
            q=q,
            qd=qd,
            tau=tau,
            tau_is_model_space=True,
        )
        if emb is not None:
            block_until_ready = getattr(emb, "block_until_ready", None)
            if block_until_ready is not None:
                block_until_ready()

    print(
        "[prepare_history_encoder_cache] complete "
        f"elapsed_s={time.perf_counter() - start:.2f} "
        f"cache_dir={Path(args.history_jax_cache_dir).expanduser().resolve()} "
        f"dof={adaptor._dof} patch_size={getattr(adaptor, '_patch_size', 'n/a')} "
        f"patch_stride={getattr(adaptor, '_patch_stride', 'n/a')} "
        f"decode_patch_size={getattr(adaptor, '_decode_patch_size', 'n/a')} "
        f"attention_history_s={args.attention_history_s} "
        f"temporal_history_patches={getattr(adaptor, '_attention_history_tokens', None)} "
        f"attention_history_cache_tokens="
        f"{getattr(adaptor, '_attention_history_cache_tokens', None)}"
    )


if __name__ == "__main__":
    main()
