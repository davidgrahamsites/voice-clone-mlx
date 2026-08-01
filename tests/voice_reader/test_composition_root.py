"""Test the Voice Reader composition root.

The root is the one place that decides which runtimes exist. It builds a
registry and hands it back; it owns no global, wires no UI, and nothing in the
package imports it — so deleting it breaks nothing but the app entry point.
"""

import ast
import io
import wave
from pathlib import Path

import pytest

from voiceclonegpt.reader_app.composition import build_runtime_registry
from voiceclonegpt.synthesis.runtime_registry import (
    RuntimeRegistrationError,
    RuntimeRegistry,
)


def artifact(tmp_path):
    """Create a real artifact file: the null runtime's `load` requires one."""
    path = tmp_path / "model.bin"
    path.write_bytes(b"")
    return path


def read_wav(payload: bytes):
    """Parse WAV bytes, failing if they are not a valid RIFF/WAVE stream."""
    with wave.open(io.BytesIO(payload), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frame_rate": handle.getframerate(),
            "frames": handle.getnframes(),
        }


class TestBuiltRegistry:
    """What the root wires up, stated as behavior rather than constants."""

    def test_registers_exactly_the_null_runtime(self):
        assert build_runtime_registry().ids() == ("null",)

    def test_returns_a_runtime_registry(self):
        assert isinstance(build_runtime_registry(), RuntimeRegistry)

    def test_registered_runtime_satisfies_the_adapter_shape(self):
        runtime = build_runtime_registry().get("null")

        assert callable(runtime.load)
        assert callable(runtime.synthesize)

    def test_registered_runtime_synthesizes_real_wav_bytes(self, tmp_path):
        """The wiring is only real if the thing it wired can actually run."""
        runtime = build_runtime_registry().get("null")

        model = runtime.load(artifact(tmp_path), {})
        info = read_wav(runtime.synthesize(model, "A line from the reader."))

        assert info["channels"] == 1
        assert info["sample_width"] == 2
        assert info["frames"] > 0
        assert info["frame_rate"] > 0

    def test_get_returns_the_same_instance_every_time(self):
        """This registry stores adapters, so callers share one object."""
        registry = build_runtime_registry()

        assert registry.get("null") is registry.get("null")


class TestBuiltRegistryRefusals:
    """The root hands back a registry that still refuses ambiguity."""

    def test_duplicate_registration_raises(self):
        registry = build_runtime_registry()

        with pytest.raises(RuntimeRegistrationError, match="already registered"):
            registry.register("null", registry.get("null"))

    def test_unknown_id_raises(self):
        with pytest.raises(RuntimeRegistrationError, match="unknown runtime"):
            build_runtime_registry().get("nope")


class TestEachBuildIsIndependent:
    """No shared global: one caller's registry cannot leak into another's."""

    def test_two_builds_are_distinct_objects(self):
        assert build_runtime_registry() is not build_runtime_registry()

    def test_registering_into_one_does_not_affect_the_next(self):
        first = build_runtime_registry()
        first.register("extra", first.get("null"))

        assert build_runtime_registry().ids() == ("null",)


class TestReadinessCanListTheBuiltRegistry:
    """Readiness asks a registry "what could be selected"; ours can answer."""

    def test_satisfies_the_lister_protocol_readiness_uses(self):
        """`readiness._registered_runtimes` picks `available` else `ids`."""
        registry = build_runtime_registry()

        lister = getattr(registry, "available", None) or getattr(registry, "ids")

        assert tuple(lister()) == ("null",)


class TestCompositionRootIsDetachable:
    """Nothing may import the root: it is a leaf the app entry point calls."""

    def test_no_module_in_the_package_imports_composition(self):
        package = Path(build_runtime_registry.__module__.split(".")[0])
        root = Path(__file__).resolve().parents[2] / "src" / "voiceclonegpt"
        importers = []

        for path in root.rglob("*.py"):
            if path.name == "composition.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                if any("composition" in name for name in names):
                    importers.append(str(path.relative_to(root)))

        assert importers == [], f"composition is imported by {importers}"
        assert package.name == "voiceclonegpt"

    def test_imports_no_ui_or_backend(self):
        import voiceclonegpt.reader_app.composition as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line.strip()
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        assert "ui" not in imports
        assert "tts" not in imports
        assert "mlx" not in imports
        assert "tkinter" not in imports
