"""FastEmbed cache validation and repair."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_validated: set[Path] = set()


def _resolve_cache_dir(cache_dir: str | None) -> Path:
    """Resolve the FastEmbed cache directory using the same precedence FastEmbed uses."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    env = os.environ.get("FASTEMBED_CACHE_PATH")
    if env:
        return Path(env).expanduser()
    return Path(tempfile.gettempdir()) / "fastembed_cache"


def _is_model_dir_broken(model_dir: Path) -> tuple[bool, str]:
    """Return (broken, reason) for a `models--*` directory."""
    blobs = model_dir / "blobs"
    if blobs.is_dir():
        incomplete = list(blobs.glob("*.incomplete"))
        if incomplete:
            return True, f"{len(incomplete)} incomplete blob(s) in {blobs}"

    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return True, f"missing snapshots dir at {snapshots}"

    snapshot_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        return True, f"no snapshots in {snapshots}"

    for snap in snapshot_dirs:
        onnx_files = [p for p in snap.iterdir() if p.suffix == ".onnx"]
        for onnx in onnx_files:
            try:
                target = onnx.resolve(strict=True)
            except (FileNotFoundError, OSError):
                return True, f"dangling .onnx symlink {onnx}"
            if target.stat().st_size == 0:
                return True, f"empty .onnx blob {target}"
        if onnx_files:
            return False, ""

    return True, f"no .onnx file in any snapshot under {snapshots}"


def validate_fastembed_cache(cache_dir: str | None) -> None:
    """Walk the FastEmbed cache and prune any broken `models--*` directories.

    Idempotent across a process: each resolved cache directory is validated only once.
    Safe to call before each model load.
    """
    root = _resolve_cache_dir(cache_dir)
    if root in _validated:
        return
    _validated.add(root)

    if not root.is_dir():
        return

    for model_dir in root.glob("models--*"):
        if not model_dir.is_dir():
            continue
        broken, reason = _is_model_dir_broken(model_dir)
        if broken:
            logger.warning(
                "FastEmbed cache entry %s is corrupted (%s); removing and forcing re-download.",
                model_dir,
                reason,
            )
            shutil.rmtree(model_dir, ignore_errors=True)
