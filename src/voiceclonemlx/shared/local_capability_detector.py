"""Detect local OS and runtime execution capabilities.

Probes are pure import checks and version reads. No model loading, no CUDA
initialization, no weights, no filesystem or network access. Only stdlib:
importlib.util, importlib.metadata, platform.
"""

import importlib.metadata
import importlib.util
import platform
from dataclasses import dataclass
from typing import Optional

__all__ = ["LocalCapabilityError", "LocalCapabilitySet", "detect_capabilities"]

class LocalCapabilityError(ValueError):
    """A capability probe failed or returned an unusable value."""

@dataclass(frozen=True)
class LocalCapabilitySet:
    """Immutable result of OS/runtime capability probes."""
    has_pytorch: bool
    has_mlx: bool
    has_cuda: bool
    has_metal: bool
    python_version: str
    torch_version: Optional[str]
    mlx_version: Optional[str]
    def __post_init__(self) -> None:
        for name in ("has_pytorch", "has_mlx", "has_cuda", "has_metal"):
            if not isinstance(getattr(self, name), bool):
                raise LocalCapabilityError(f"{name} must be a bool")
        if not isinstance(self.python_version, str) or not self.python_version.strip():
            raise LocalCapabilityError(
                f"python_version must be a non-empty string: {self.python_version!r}"
            )
        for name in ("torch_version", "mlx_version"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise LocalCapabilityError(f"{name} must be string or None: {v!r}")
        if self.has_cuda and not self.has_pytorch:
            raise LocalCapabilityError("has_cuda=True requires has_pytorch=True")
        if self.has_metal and not self.has_pytorch:
            raise LocalCapabilityError("has_metal=True requires has_pytorch=True")
        if self.torch_version is not None and not self.has_pytorch:
            raise LocalCapabilityError("torch_version set but has_pytorch=False")
        if self.mlx_version is not None and not self.has_mlx:
            raise LocalCapabilityError("mlx_version set but has_mlx=False")

def _get_version(package: str) -> Optional[str]:
    """Read version from package metadata without importing."""
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None

def detect_capabilities() -> LocalCapabilitySet:
    """Probe OS and installed packages; report available capabilities.

    Probes are pure import spec checks; no initialization or API calls.
    Hardware flags are conservative (false unless explicitly verified).
    Versions come from package metadata, not by importing.
    """
    has_pytorch = importlib.util.find_spec("torch") is not None
    has_mlx = importlib.util.find_spec("mlx") is not None
    torch_version = _get_version("torch") if has_pytorch else None
    mlx_version = _get_version("mlx") if has_mlx else None
    return LocalCapabilitySet(
        has_pytorch=has_pytorch,
        has_mlx=has_mlx,
        has_cuda=False,
        has_metal=False,
        python_version=platform.python_version(),
        torch_version=torch_version,
        mlx_version=mlx_version,
    )
