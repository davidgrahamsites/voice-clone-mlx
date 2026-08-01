"""Synthesized audio must land where the Reader already looks for it.

The Reader's discovery rule is `reader_app.core.find_audio` — it is imported
here and not restated, so this test fails if the two ever drift apart.
"""

import ast
import json
from pathlib import Path

import pytest

from voiceclonegpt.reader_app.core import find_audio, load_manifest
from voiceclonegpt.tts.backend import FakeBackend

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "voiceclonegpt"

MANIFEST_ROWS = [
    {"id": "script_session_1_0001", "text": "The first line.", "style": "neutral"},
    {"id": "script_session_1_0002", "text": "The second line.", "style": "warm"},
]


@pytest.fixture
def session(tmp_path):
    """A manifest beside a reference clip, the way the Studio writes one."""
    manifest_path = tmp_path / "script_session_1.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in MANIFEST_ROWS) + "\n", encoding="utf-8"
    )
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"reference clip")
    return manifest_path, ref_audio


class TestRoundTrip:
    def test_synthesized_audio_is_discovered_by_the_reader(self, tmp_path, session):
        manifest_path, ref_audio = session
        backend = FakeBackend()

        for row in MANIFEST_ROWS:
            backend.synthesize(
                row["text"], ref_audio, manifest_path.parent / f"{row['id']}.wav"
            )

        for row in MANIFEST_ROWS:
            found = find_audio(manifest_path.parent, row["id"])
            assert found is not None, f"reader did not find audio for {row['id']}"
            assert found.name == f"{row['id']}.wav"

    def test_every_take_loads_with_audio(self, session):
        manifest_path, ref_audio = session
        backend = FakeBackend()

        for row in MANIFEST_ROWS:
            backend.synthesize(
                row["text"], ref_audio, manifest_path.parent / f"{row['id']}.wav"
            )

        takes = load_manifest(manifest_path)

        assert len(takes) == len(MANIFEST_ROWS)
        assert all(take.has_audio for take in takes)

    def test_unsynthesized_utterances_stay_without_audio(self, session):
        manifest_path, ref_audio = session

        FakeBackend().synthesize(
            MANIFEST_ROWS[0]["text"],
            ref_audio,
            manifest_path.parent / f"{MANIFEST_ROWS[0]['id']}.wav",
        )

        takes = load_manifest(manifest_path)

        assert takes[0].has_audio
        assert not takes[1].has_audio


class TestIsolation:
    """`tts` is a leaf: it may be deleted without breaking anything else."""

    def test_no_module_outside_tts_imports_tts(self):
        offenders = []
        for path in SRC_ROOT.rglob("*.py"):
            if "tts" in path.relative_to(SRC_ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    hit = any(_names_tts(a.name, 0) for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    hit = _names_tts(node.module or "", node.level) or (
                        node.level > 0
                        and not node.module
                        and any(a.name == "tts" for a in node.names)
                    )
                else:
                    continue
                if hit:
                    offenders.append(str(path.relative_to(SRC_ROOT)))

        assert offenders == [], f"tts must stay unwired; imported by {offenders}"

    def test_the_scan_would_notice_an_import(self, tmp_path):
        """The check above is only worth having if it can fail."""
        assert _names_tts("voiceclonegpt.tts.backend", 0)
        assert _names_tts("tts.backend", 1)
        # mlx-audio's own `tts` subpackage is a different thing entirely.
        assert not _names_tts("mlx_audio.tts.utils", 0)


def _names_tts(module_name: str, level: int) -> bool:
    """True if an import target is *this package's* `tts` package.

    Absolute imports must be `voiceclonegpt.tts...`; relative ones (`level`
    above zero) are already inside the package, so a leading `tts` is enough.
    Third-party modules with a `tts` submodule — mlx-audio has one — are not
    this package and must not be flagged.
    """
    parts = module_name.split(".") if module_name else []
    if level > 0:
        return parts[:1] == ["tts"]
    return parts[:2] == ["voiceclonegpt", "tts"]
