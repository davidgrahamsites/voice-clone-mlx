"""Test OS/runtime capability probes."""
import dataclasses
from pathlib import Path
import pytest
from voiceclonemlx.shared.local_capability_detector import (
    LocalCapabilityError, LocalCapabilitySet, detect_capabilities,
)

def test_detect_returns_frozen_set():
    result = detect_capabilities()
    assert isinstance(result, LocalCapabilitySet)
    assert dataclasses.is_dataclass(result)

def test_all_fields_typed_correctly():
    result = detect_capabilities()
    assert isinstance(result.has_pytorch, bool)
    assert isinstance(result.has_mlx, bool)
    assert isinstance(result.has_cuda, bool)
    assert isinstance(result.has_metal, bool)
    assert result.python_version and isinstance(result.python_version, str)
    assert result.torch_version is None or isinstance(result.torch_version, str)
    assert result.mlx_version is None or isinstance(result.mlx_version, str)
def test_result_is_frozen():
    result = detect_capabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.has_pytorch = not result.has_pytorch

@pytest.mark.parametrize("bad", ["", "   ", None, 42, True])
def test_python_version_required_nonempty_string(bad):
    with pytest.raises(LocalCapabilityError, match="python_version"):
        LocalCapabilitySet(
            has_pytorch=False, has_mlx=False, has_cuda=False, has_metal=False,
            python_version=bad, torch_version=None, mlx_version=None,
        )

@pytest.mark.parametrize("bad", ["", "   ", 42, True])
def test_version_strings_or_none(bad):
    with pytest.raises(LocalCapabilityError):
        LocalCapabilitySet(
            has_pytorch=False, has_mlx=False, has_cuda=False, has_metal=False,
            python_version="3.11", torch_version=bad, mlx_version=None,
        )

@pytest.mark.parametrize("flag", ["has_pytorch", "has_mlx", "has_cuda", "has_metal"])
def test_bool_flags_required(flag):
    kwargs={"has_pytorch":False,"has_mlx":False,"has_cuda":False,"has_metal":False,
            "python_version":"3.11","torch_version":None,"mlx_version":None}
    kwargs[flag] = "true"
    with pytest.raises(LocalCapabilityError, match=flag):
        LocalCapabilitySet(**kwargs)

@pytest.mark.parametrize("hw,pytorch_flag", [("cuda", "has_cuda"), ("metal", "has_metal")])
def test_hw_requires_pytorch(hw, pytorch_flag):
    kwargs = {"has_pytorch": False, "has_mlx": False, "has_cuda": False,
              "has_metal": False, "python_version": "3.11", "torch_version": None,
              "mlx_version": None}
    kwargs[pytorch_flag] = True
    with pytest.raises(LocalCapabilityError, match="has_pytorch"):
        LocalCapabilitySet(**kwargs)

@pytest.mark.parametrize("ver_field,pkg", [("torch_version", "pytorch"), ("mlx_version", "mlx")])
def test_version_requires_package(ver_field, pkg):
    kwargs={"has_pytorch":False,"has_mlx":False,"has_cuda":False,"has_metal":False,
            "python_version":"3.11","torch_version":None,"mlx_version":None}
    kwargs[ver_field] = "1.0"
    with pytest.raises(LocalCapabilityError, match=ver_field):
        LocalCapabilitySet(**kwargs)
