"""Packaged robot assets and downloadable checkpoints.

``panda_pandagripper.xml`` (plus its meshes) is the exact ideal-model MJCF the
released Panda checkpoints were trained against; it is installed with the
package so deployments need no separate asset download.

Checkpoints are larger and are fetched on demand from the project's GitHub
releases into a local cache (``~/.cache/simadaptor/checkpoints`` by default,
override with the ``SIMADAPTOR_CACHE_DIR`` environment variable)::

    from simadaptor.assets import default_panda_xml, fetch_checkpoint

    ckpt = fetch_checkpoint()            # downloads once, cached afterwards
    xml = default_panda_xml()
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PANDA_PANDAGRIPPER_XML = Path(__file__).resolve().parent / "franka_panda" / "panda_pandagripper.xml"

_RELEASE_BASE = "https://github.com/Dongwon-Son/TAM/releases/download"

# name -> (release tag, tarball, sha256 of the tarball, checkpoint dir inside)
CHECKPOINTS: dict[str, tuple[str, str, str, str]] = {
    # Panda-specific TAM, DAgger-finetuned on a real Panda (default).
    "dagger_applied_8850": (
        "checkpoints-v1",
        "dagger_applied_8850.tar.gz",
        "43fd2715034dd2b9ad86cca5a2f0b1d1b8a4f5948f53395c709159b50be5789e",
        "dagger_applied_8850/checkpoint_8850",
    ),
    # Panda-specific TAM, simulation-trained only (the pre-DAgger model).
    "panda_specific": (
        "checkpoints-v1",
        "panda_specific.tar.gz",
        "adb6ccccabfa6ba3c733002519a5e049ea621c7da72c1584f9dac010550b1d4b",
        "panda_specific/checkpoint_960000",
    ),
}

DEFAULT_CHECKPOINT = "dagger_applied_8850"


def default_panda_xml() -> Path:
    """The packaged ideal-model MJCF (Panda + Franka hand) used in training."""
    if not PANDA_PANDAGRIPPER_XML.is_file():
        raise FileNotFoundError(f"packaged ideal-model XML missing: {PANDA_PANDAGRIPPER_XML}")
    return PANDA_PANDAGRIPPER_XML


def _cache_root(cache_dir: str | os.PathLike | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("SIMADAPTOR_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "simadaptor" / "checkpoints"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (dest / member.name).resolve()
            if not str(member_path).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(dest)


def fetch_checkpoint(
    name: str = DEFAULT_CHECKPOINT,
    cache_dir: str | os.PathLike | None = None,
) -> Path:
    """Return a local checkpoint directory, downloading it on first use.

    The returned path points at the ``checkpoint_<step>`` directory and can be
    passed directly as ``simadaptor_ckpt_path``. Downloads are verified against
    a pinned SHA-256 and cached; delete the cache directory to force a
    re-download.
    """
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; available: {sorted(CHECKPOINTS)}")
    tag, filename, sha256, ckpt_subdir = CHECKPOINTS[name]
    root = _cache_root(cache_dir)
    ckpt_path = root / ckpt_subdir
    if ckpt_path.is_dir():
        return ckpt_path

    url = f"{_RELEASE_BASE}/{tag}/{filename}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"[simadaptor.assets] downloading {name} from {url} ...")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tar_path = Path(tmp) / filename
        with urllib.request.urlopen(url) as resp, open(tar_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        digest = _sha256(tar_path)
        if digest != sha256:
            raise RuntimeError(
                f"checksum mismatch for {filename}: expected {sha256}, got {digest}"
            )
        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir()
        _safe_extract(tar_path, extract_dir)
        bundle_root = extract_dir / Path(ckpt_subdir).parts[0]
        if not bundle_root.is_dir():
            raise RuntimeError(f"archive did not contain {Path(ckpt_subdir).parts[0]!r}")
        final = root / Path(ckpt_subdir).parts[0]
        if not final.exists():
            os.replace(bundle_root, final)
    if not ckpt_path.is_dir():
        raise RuntimeError(f"checkpoint directory missing after extraction: {ckpt_path}")
    print(f"[simadaptor.assets] cached at {ckpt_path}")
    return ckpt_path


__all__ = [
    "CHECKPOINTS",
    "DEFAULT_CHECKPOINT",
    "PANDA_PANDAGRIPPER_XML",
    "default_panda_xml",
    "fetch_checkpoint",
]
