import importlib.util
import sys
from pathlib import Path

from simadaptor.cli import parse_tyro_config

_IMPL_MODULE = None


def _load_impl_module():
    global _IMPL_MODULE
    if _IMPL_MODULE is not None:
        return _IMPL_MODULE
    impl_path = Path(__file__).with_name("dagger_finetune_impl.py")
    module_name = f"{__name__.replace('.', '_')}_impl"
    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load implementation module from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    _IMPL_MODULE = module
    return module


def main(cfg) -> None:
    _load_impl_module().main(cfg)


def __getattr__(name: str):
    return getattr(_load_impl_module(), name)


if __name__ == "__main__":
    cfg = parse_tyro_config(_load_impl_module().OnlineDaggerConfig)
    main(cfg)
