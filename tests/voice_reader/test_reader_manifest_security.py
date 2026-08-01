"""Manifest schema and audio-path security for Voice Reader.

A recording manifest is untrusted input: it may be hand-edited, copied in, or
produced by another tool. These tests pin what `load_manifest` accepts and
prove `find_audio` cannot be steered outside the chosen audio directory.

No Tk, no display, no opt-in needed — everything here is pure logic and
temporary files. Player and window behaviour live in `test_reader_app.py`.
"""

import json

import pytest

from voiceclonegpt.reader_app.core import find_audio, load_manifest


@pytest.fixture
def manifest(tmp_path):
    records = [
        {"id": "NEUTRAL-001", "text": "First line.", "style": "neutral"},
        {"id": "NEUTRAL-002", "text": "Second line.", "style": "neutral"},
    ]
    path = tmp_path / "script_session_1.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (tmp_path / "NEUTRAL-001.wav").write_bytes(b"")
    return path


class TestLoadManifest:
    def test_pairs_takes_with_audio_found_next_to_the_manifest(self, manifest):
        takes = load_manifest(manifest)
        assert [t.id for t in takes] == ["NEUTRAL-001", "NEUTRAL-002"]
        assert takes[0].has_audio
        assert not takes[1].has_audio

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.jsonl")

    def test_malformed_line_reports_its_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"id": "A", "text": "Hi.", "style": "neutral"}\nnot json\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=":2"):
            load_manifest(path)

    def test_audio_dir_can_be_separate(self, manifest, tmp_path):
        empty = tmp_path / "elsewhere"
        empty.mkdir()
        assert all(not t.has_audio for t in load_manifest(manifest, empty))

    def test_find_audio_returns_none_for_blank_id(self, tmp_path):
        assert find_audio(tmp_path, "") is None


class TestManifestSchema:
    """Every row must be an object with string id, text, and style."""

    def _write(self, tmp_path, line):
        path = tmp_path / "schema.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        "line",
        ['["not", "an", "object"]', '"a string"', "42", "null", "true"],
    )
    def test_rejects_rows_that_are_not_objects(self, tmp_path, line):
        with pytest.raises(ValueError, match="not a JSON object"):
            load_manifest(self._write(tmp_path, line))

    @pytest.mark.parametrize("field", ["id", "text", "style"])
    def test_rejects_missing_required_field(self, tmp_path, field):
        record = {"id": "A-1", "text": "Hi.", "style": "neutral"}
        del record[field]

        with pytest.raises(ValueError, match=field):
            load_manifest(self._write(tmp_path, json.dumps(record)))

    @pytest.mark.parametrize("field", ["id", "text", "style"])
    def test_rejects_non_string_field(self, tmp_path, field):
        record = {"id": "A-1", "text": "Hi.", "style": "neutral"}
        record[field] = 17

        with pytest.raises(ValueError, match=field):
            load_manifest(self._write(tmp_path, json.dumps(record)))

    def test_rejects_empty_id(self, tmp_path):
        record = {"id": "", "text": "Hi.", "style": "neutral"}

        with pytest.raises(ValueError, match="id"):
            load_manifest(self._write(tmp_path, json.dumps(record)))

    def test_rejects_unsafe_id(self, tmp_path):
        record = {"id": "../escape", "text": "Hi.", "style": "neutral"}

        with pytest.raises(ValueError, match="id"):
            load_manifest(self._write(tmp_path, json.dumps(record)))

    def test_schema_error_reports_its_line_number(self, tmp_path):
        path = tmp_path / "schema.jsonl"
        path.write_text(
            json.dumps({"id": "A-1", "text": "Hi.", "style": "neutral"}) + "\n"
            + json.dumps({"id": "A-2", "text": 5, "style": "neutral"}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=":2"):
            load_manifest(path)

    def test_accepts_a_well_formed_row(self, tmp_path):
        record = {"id": "A-1", "text": "Hi.", "style": "neutral"}

        takes = load_manifest(self._write(tmp_path, json.dumps(record)))

        assert takes[0].id == "A-1"


class TestFindAudioIsConfined:
    """A manifest is untrusted input; it must not reach outside audio_dir."""

    @pytest.mark.parametrize(
        "utterance_id",
        [
            "../secret",
            "../../etc/passwd",
            "sub/nested",
            "sub\\nested",
            "/etc/passwd",
            "..",
            ".",
            "   ",
            "with\x00null",
        ],
    )
    def test_rejects_unsafe_ids(self, tmp_path, utterance_id):
        assert find_audio(tmp_path, utterance_id) is None

    def test_traversal_cannot_reach_a_real_file_outside(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.wav").write_bytes(b"")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        assert find_audio(audio_dir, "../outside/secret") is None

    def test_symlink_escaping_the_audio_dir_is_rejected(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "real.wav"
        target.write_bytes(b"")

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "ESCAPE-1.wav").symlink_to(target)

        assert find_audio(audio_dir, "ESCAPE-1") is None

    def test_symlink_inside_the_audio_dir_is_allowed(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        real = audio_dir / "REAL-1.wav"
        real.write_bytes(b"")
        (audio_dir / "LINK-1.wav").symlink_to(real)

        found = find_audio(audio_dir, "LINK-1")

        # The resolved target is returned, and it is inside audio_dir.
        assert found == real.resolve()
        assert found.relative_to(audio_dir.resolve())

    def test_returns_the_resolved_path(self, tmp_path):
        """Callers act on the checked path, not one that could re-resolve."""
        (tmp_path / "OK-1.wav").write_bytes(b"")

        found = find_audio(tmp_path, "OK-1")

        assert found == found.resolve()
        assert found.is_absolute()

    def test_resolved_path_stays_inside_a_symlinked_audio_dir(self, tmp_path):
        real_dir = tmp_path / "real_audio"
        real_dir.mkdir()
        (real_dir / "OK-1.wav").write_bytes(b"")
        link_dir = tmp_path / "linked_audio"
        link_dir.symlink_to(real_dir)

        found = find_audio(link_dir, "OK-1")

        assert found == (real_dir / "OK-1.wav").resolve()

    def test_directory_named_like_audio_is_not_returned(self, tmp_path):
        (tmp_path / "DIR-1.wav").mkdir()

        assert find_audio(tmp_path, "DIR-1") is None

    def test_finds_a_normal_file(self, tmp_path):
        (tmp_path / "OK-1.wav").write_bytes(b"")

        assert find_audio(tmp_path, "OK-1") == (tmp_path / "OK-1.wav")

    def test_works_when_the_audio_dir_is_itself_a_symlink(self, tmp_path):
        real_dir = tmp_path / "real_audio"
        real_dir.mkdir()
        (real_dir / "OK-1.wav").write_bytes(b"")
        link_dir = tmp_path / "linked_audio"
        link_dir.symlink_to(real_dir)

        assert find_audio(link_dir, "OK-1") is not None

    def test_missing_audio_dir_is_not_an_error(self, tmp_path):
        assert find_audio(tmp_path / "absent", "OK-1") is None

