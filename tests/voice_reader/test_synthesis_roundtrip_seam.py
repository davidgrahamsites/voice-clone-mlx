"""Boundary tests for the lazy provider-neutral round-trip import."""

from pathlib import Path

import pytest

from voiceclonemlx.reader_app import synthesis_session
from voiceclonemlx.reader_app.synthesis_session import (
    ReaderSynthesisError,
    ReaderSynthesisSession,
)


@pytest.fixture
def bundle_dir(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    return path


@pytest.fixture
def output_dir(tmp_path):
    path = tmp_path / "out"
    path.mkdir()
    return path


class TestRoundTripSeam:
    """The default round trip is the shared provider-neutral contract."""

    def test_default_roundtrip_errors_are_wrapped(self, bundle_dir, output_dir):
        session = ReaderSynthesisSession(
            bundle_dir=bundle_dir,
            runtime_id="mlx",
            runtime=object(),
            output_dir=output_dir,
        )

        with pytest.raises(ReaderSynthesisError) as exc:
            session.synthesize("Hi.", "a.wav")

        assert isinstance(exc.value, ReaderSynthesisError)
        assert exc.value.__cause__ is not None

    def test_module_does_not_import_roundtrip_at_module_scope(self):
        source = Path(synthesis_session.__file__).read_text(encoding="utf-8")
        imports = [
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ]

        assert not any("roundtrip" in line for line in imports)

    def test_module_reaches_no_network(self):
        source = Path(synthesis_session.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("urllib", "requests", "socket", "subprocess", "http"):
            assert banned not in imports

    def test_module_imports_no_ui(self):
        source = Path(synthesis_session.__file__).read_text(encoding="utf-8")

        assert "tkinter" not in source
        assert "from .ui" not in source
