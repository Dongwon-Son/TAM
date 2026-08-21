from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_DEPLOY_JAX_CACHE_XLA_CACHES = "xla_gpu_per_fusion_autotune_cache_dir"
_DISABLED_CACHE_TOKENS = {"", "0", "false", "none", "off", "disabled"}


def normalize_jax_cache_dir(cache_dir: str | Path | None) -> Path | None:
    if cache_dir is None:
        return None
    raw = str(cache_dir).strip()
    if raw.lower() in _DISABLED_CACHE_TOKENS:
        return None
    return Path(raw).expanduser().resolve()


def normalize_jax_cache_xla_caches(xla_caches: str | None) -> str | None:
    if xla_caches is None:
        return None
    raw = str(xla_caches).strip()
    if raw.lower() in _DISABLED_CACHE_TOKENS:
        return None
    return raw


def _env_bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _try_update_jax_config(name: str, value: Any, *, log_prefix: str) -> bool:
    try:
        import jax

        jax.config.update(name, value)
        return True
    except Exception as exc:
        print(f"{log_prefix} Warning: failed to set JAX config {name}={value!r}: {exc}")
        return False


def configure_jax_persistent_cache(
    cache_dir: str | Path | None,
    *,
    min_compile_time_s: float = 0.0,
    min_entry_size_bytes: int = -1,
    xla_caches: str | None = DEFAULT_DEPLOY_JAX_CACHE_XLA_CACHES,
    log_prefix: str = "[jax_cache]",
) -> Path | None:
    """
    Enable JAX's persistent compilation cache before deploy-time JITs compile.

    ``min_compile_time_s=0`` and ``min_entry_size_bytes=-1`` are deploy-friendly:
    they ask JAX to persist even small executables, which helps repeated short
    startup/warmup runs hit the cache instead of recompiling.
    """
    if isinstance(cache_dir, str) and cache_dir == "auto":
        # Default persistent-cache location so repeated deploy runs skip the
        # multi-second JIT warm-up. SIMADAPTOR_JAX_CACHE_DIR overrides; an
        # empty value disables caching.
        env = os.environ.get("SIMADAPTOR_JAX_CACHE_DIR")
        if env is not None:
            cache_dir = env or None
        else:
            cache_dir = Path.home() / ".cache" / "simadaptor" / "jax"
    cache_path = normalize_jax_cache_dir(cache_dir)
    if cache_path is None:
        return None

    cache_path.mkdir(parents=True, exist_ok=True)
    xla_caches = normalize_jax_cache_xla_caches(xla_caches)

    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = _env_bool_text(True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(float(min_compile_time_s))
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(int(min_entry_size_bytes))
    if xla_caches:
        os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = str(xla_caches)

    _try_update_jax_config("jax_enable_compilation_cache", True, log_prefix=log_prefix)
    cache_ok = _try_update_jax_config(
        "jax_compilation_cache_dir",
        str(cache_path),
        log_prefix=log_prefix,
    )
    if not cache_ok:
        try:
            from jax.experimental.compilation_cache import compilation_cache

            compilation_cache.set_cache_dir(str(cache_path))
            cache_ok = True
        except Exception as exc:
            print(f"{log_prefix} Warning: failed to enable JAX compilation cache: {exc}")
    _try_update_jax_config(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_time_s),
        log_prefix=log_prefix,
    )
    _try_update_jax_config(
        "jax_persistent_cache_min_entry_size_bytes",
        int(min_entry_size_bytes),
        log_prefix=log_prefix,
    )
    if xla_caches:
        _try_update_jax_config(
            "jax_persistent_cache_enable_xla_caches",
            str(xla_caches),
            log_prefix=log_prefix,
        )
    if cache_ok:
        print(
            f"{log_prefix} JAX persistent compilation cache enabled at {cache_path} "
            f"(min_compile_time_s={float(min_compile_time_s):g}, "
            f"min_entry_size_bytes={int(min_entry_size_bytes)}, "
            f"xla_caches={xla_caches or 'disabled'})."
        )
    return cache_path
