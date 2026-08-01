import json
import tempfile
from pathlib import Path
import pytest

from voiceclonemlx.recording.script_generator import (
    generate_script,
    ScriptOutput,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

class TestRendererSeam:
    """The renderer turns utterances into Markdown and nothing else."""

    def _utterances(self):
        from voiceclonemlx.recording.script_generator import Utterance

        return [
            Utterance(
                id="NEUTRAL-aaaa1111",
                text="First line.",
                style="neutral",
                source="/tmp/source.txt",
                session=3,
                block=1,
            ),
            Utterance(
                id="WARM-bbbb2222",
                text="# Not a heading",
                style="warm",
                source="/tmp/source.txt",
                session=3,
                block=2,
            ),
        ]

    def test_render_script_is_importable_and_pure(self):
        """Rendering returns a string and writes nothing."""
        from voiceclonemlx.recording.script_rendering import render_script

        markdown = render_script(self._utterances(), ["neutral", "warm"], 3)

        assert isinstance(markdown, str)
        assert "Session 3" in markdown
        assert "NEUTRAL-aaaa1111" in markdown

    def test_render_script_emits_one_section_per_style(self):
        """Each requested style with utterances gets its own heading."""
        from voiceclonemlx.recording.script_rendering import render_script

        markdown = render_script(self._utterances(), ["neutral", "warm"], 3)
        headings = [
            line for line in markdown.splitlines() if line.startswith("##")
        ]

        assert headings == ["## Neutral Reading", "## Warm Reading"]

    def test_render_script_skips_styles_without_utterances(self):
        """A style with no utterances produces no section."""
        from voiceclonemlx.recording.script_rendering import render_script

        markdown = render_script(
            self._utterances(), ["neutral", "warm", "somber"], 3
        )

        assert "Somber Reading" not in markdown

    def test_escape_markdown_neutralizes_structure(self):
        """The escaping seam is directly testable."""
        from voiceclonemlx.recording.script_rendering import escape_markdown

        assert escape_markdown("# heading") == "\\# heading"
        assert escape_markdown("**bold**") == "\\*\\*bold\\*\\*"
        assert escape_markdown("a\nb") == "a b"
        assert escape_markdown("back\\slash") == "back\\\\slash"


class TestStoreSeam:
    """The store writes the JSONL manifest and nothing else."""

    def _utterances(self):
        from voiceclonemlx.recording.script_generator import Utterance

        return [
            Utterance(
                id="NEUTRAL-aaaa1111",
                text="First line.",
                style="neutral",
                source="/tmp/source.txt",
                session=3,
                block=1,
            )
        ]

    def test_write_manifest_writes_one_json_object_per_line(self, temp_dir):
        """Manifest is JSONL with the raw, unescaped text."""
        from voiceclonemlx.recording.script_storage import write_manifest

        path = temp_dir / "manifest.jsonl"
        write_manifest(path, self._utterances())

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["id"] == "NEUTRAL-aaaa1111"
        assert entry["text"] == "First line."
        assert entry["session"] == 3

    def test_write_manifest_overwrites_previous_content(self, temp_dir):
        """Rewriting a manifest replaces it rather than appending."""
        from voiceclonemlx.recording.script_storage import write_manifest

        path = temp_dir / "manifest.jsonl"
        write_manifest(path, self._utterances())
        write_manifest(path, self._utterances())

        assert len(path.read_text().strip().split("\n")) == 1

    def test_write_script_markdown_writes_the_file(self, temp_dir):
        """Markdown writing is a storage responsibility, not the renderer's."""
        from voiceclonemlx.recording.script_storage import write_script_markdown

        path = temp_dir / "script.md"
        write_script_markdown(path, "# Recording Script\nCafé 🎤\n")

        assert path.read_text(encoding="utf-8") == "# Recording Script\nCafé 🎤\n"

    def test_both_outputs_are_written_through_storage(self, temp_dir, monkeypatch):
        """The coordinator must not write either output file directly."""
        from voiceclonemlx.recording import script_generator

        calls = []

        def fake_markdown(path, content):
            calls.append(("markdown", Path(path)))

        def fake_manifest(path, utterances):
            calls.append(("manifest", Path(path)))

        monkeypatch.setattr(
            script_generator, "write_script_markdown", fake_markdown
        )
        monkeypatch.setattr(script_generator, "write_manifest", fake_manifest)

        source_file = temp_dir / "source.txt"
        source_file.write_text("Only sentence.")

        output = script_generator.generate_script(
            source_files=[source_file], output_dir=temp_dir
        )

        assert [kind for kind, _ in calls] == ["markdown", "manifest"]
        assert dict(calls)["markdown"] == output.markdown_path
        assert dict(calls)["manifest"] == output.manifest_path

        # With both seams stubbed out, nothing should have reached disk
        assert not output.markdown_path.exists()
        assert not output.manifest_path.exists()

    def test_write_manifest_preserves_unicode(self, temp_dir):
        """Non-ASCII text round-trips through the manifest."""
        from voiceclonemlx.recording.script_generator import Utterance
        from voiceclonemlx.recording.script_storage import write_manifest

        path = temp_dir / "manifest.jsonl"
        write_manifest(
            path,
            [
                Utterance(
                    id="NEUTRAL-cccc3333",
                    text="Café naïve 🎤",
                    style="neutral",
                    source="/tmp/source.txt",
                    session=1,
                    block=1,
                )
            ],
        )

        entry = json.loads(path.read_text().strip())
        assert entry["text"] == "Café naïve 🎤"


class TestStableIds:
    """Test that utterance IDs are derived from content and source."""

    def test_ids_are_content_derived(self, temp_dir):
        """Different sentence text should yield a different ID."""
        source_a = temp_dir / "a" / "source.txt"
        source_a.parent.mkdir()
        source_a.write_text("First sentence.")

        source_b = temp_dir / "b" / "source.txt"
        source_b.parent.mkdir()
        source_b.write_text("Totally different sentence.")

        id_a = generate_script(
            source_files=[source_a], output_dir=temp_dir / "a"
        ).utterances[0].id
        id_b = generate_script(
            source_files=[source_b], output_dir=temp_dir / "b"
        ).utterances[0].id

        assert id_a != id_b

    def test_ids_survive_insertion_of_earlier_sentence(self, temp_dir):
        """Adding a sentence at the front must not renumber later utterances."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Second sentence. Third sentence.")
        before = generate_script(
            source_files=[source_file], output_dir=temp_dir
        )
        ids_before = {u.text: u.id for u in before.utterances}

        source_file.write_text(
            "A brand new opener. Second sentence. Third sentence."
        )
        after = generate_script(source_files=[source_file], output_dir=temp_dir)
        ids_after = {u.text: u.id for u in after.utterances}

        assert ids_after["Second sentence."] == ids_before["Second sentence."]
        assert ids_after["Third sentence."] == ids_before["Third sentence."]

    def test_ids_differ_by_style(self, temp_dir):
        """The same sentence in two styles should get distinct IDs."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Same sentence.")

        output = generate_script(
            source_files=[source_file],
            output_dir=temp_dir,
            styles=["neutral", "warm"],
        )

        ids = [u.id for u in output.utterances]
        assert len(set(ids)) == len(ids)

    def test_repeated_sentence_gets_distinct_ids(self, temp_dir):
        """Identical repeated text must not collide into one ID."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Repeat me. Repeat me. Repeat me.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        ids = [u.id for u in output.utterances]
        assert len(ids) == 3
        assert len(set(ids)) == 3

    def test_ids_are_stable_across_directories(self, temp_dir):
        """Same filename and content in another directory yields same IDs."""
        first = temp_dir / "one" / "source.txt"
        first.parent.mkdir()
        first.write_text("It was good to hear from you.")

        second = temp_dir / "two" / "source.txt"
        second.parent.mkdir()
        second.write_text("It was good to hear from you.")

        ids_first = [
            u.id
            for u in generate_script(
                source_files=[first], output_dir=temp_dir / "one"
            ).utterances
        ]
        ids_second = [
            u.id
            for u in generate_script(
                source_files=[second], output_dir=temp_dir / "two"
            ).utterances
        ]

        assert ids_first == ids_second

    def test_ids_differ_by_source_name(self, temp_dir):
        """Same text from differently named sources should not share IDs."""
        first = temp_dir / "letter.txt"
        first.write_text("Shared sentence.")
        second = temp_dir / "memo.txt"
        second.write_text("Shared sentence.")

        output = generate_script(
            source_files=[first, second], output_dir=temp_dir
        )

        ids = [u.id for u in output.utterances]
        assert len(set(ids)) == 2


class TestMarkdownSafety:
    """Untrusted source text must not restructure the Markdown script."""

    ADVERSARIAL = (
        "Normal opening sentence. "
        "# Injected Heading and **bold** and `code` and [link](http://evil). "
        "Ends with a pipe | and a tilde ~ here."
    )

    def _markdown(self, temp_dir, text):
        source_file = temp_dir / "source.txt"
        source_file.write_text(text)
        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        return output.markdown_path.read_text()

    def test_source_text_cannot_create_headings(self, temp_dir):
        """Only generator-authored headings may appear."""
        markdown = self._markdown(temp_dir, self.ADVERSARIAL)

        headings = [
            line for line in markdown.splitlines() if line.startswith("#")
        ]
        assert headings == ["# Recording Script", "## Neutral Reading"]

    def test_source_text_markup_is_escaped(self, temp_dir):
        """Emphasis, code, and link syntax from the source is neutralized."""
        markdown = self._markdown(temp_dir, self.ADVERSARIAL)

        assert "**bold**" not in markdown
        assert "`code`" not in markdown
        assert "[link](http://evil)" not in markdown
        assert "Injected Heading" in markdown  # text preserved, markup is not

    def test_newlines_in_source_cannot_break_the_line(self, temp_dir):
        """A newline inside a sentence must not start a new Markdown block."""
        markdown = self._markdown(
            temp_dir, "Opening line.\n## Injected block\nstill same sentence."
        )

        assert "## Injected block" not in markdown
        assert "Injected block" in markdown

    def test_utterance_lines_stay_one_per_utterance(self, temp_dir):
        """Each utterance renders as exactly one bolded ID line."""
        markdown = self._markdown(temp_dir, self.ADVERSARIAL)

        id_lines = [
            line for line in markdown.splitlines() if line.startswith("**")
        ]
        assert len(id_lines) == 3
