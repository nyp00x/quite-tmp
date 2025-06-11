from __future__ import annotations

from pathlib import Path

import base64
import shutil


from quite.io.common import uid


def to_base64(data: bytes) -> str:
    """Convert bytes to base64 string"""
    return base64.b64encode(data).decode("utf-8")


def from_base64(data: str) -> bytes:
    """Convert base64 string to bytes"""
    return base64.b64decode(data.encode("utf-8"))


def tmp_dir() -> Path:
    """Create a temporary cache directory"""
    d = Path("/tmp/quite_cache") / uid()
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_temp_file(file: bytes, file_name: str = "file.bin", directory: Path | None = None) -> Path:
    """Save a file to a temporary directory"""
    if directory is None:
        p = tmp_dir() / file_name
    else:
        p = directory / file_name
    try:
        with open(p, "wb") as f:
            f.write(file)
    except Exception as e:
        raise ValueError(f"cannot save temp file {file_name}: {e}")
    return p


def clear_tmp():
    """Clear the cache directory"""
    quite_cache = Path("/tmp/quite_cache")
    if quite_cache.exists():
        shutil.rmtree(quite_cache)
    quite_cache.mkdir(parents=True, exist_ok=True)
