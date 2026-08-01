import json
import tempfile
from pathlib import Path
import pytest

from voiceclonegpt.recording.script_generator import (
    generate_script,
    ScriptOutput,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestScriptGeneratorBasicFlow:
    """Test the basic script generation flow."""

    def test_generate_script_from_txt_file(self, temp_dir):
        """Generate a script from a simple .txt source file."""
        # Create a source file
        source_file = temp_dir / "source.txt"
        source_file.write_text(
            "It was good to hear from you after such a long and eventful week. "
            "Take all the time you need; there is no hurry at all."
        )

        # Generate script
        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        assert output is not None
        assert isinstance(output, ScriptOutput)
        assert output.markdown_path.exists()
        assert output.manifest_path.exists()

    def test_generate_script_from_docx_file(self, temp_dir):
        """Generate a script from a .docx source file."""
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph(
            "It was good to hear from you after such a long and eventful week."
        )
        document.add_paragraph("Take all the time you need; there is no hurry at all.")
        source_file = temp_dir / "source.docx"
        document.save(source_file)

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        assert output.markdown_path.exists()
        assert output.manifest_path.exists()

        manifest_lines = output.manifest_path.read_text().strip().split("\n")
        texts = [json.loads(line)["text"] for line in manifest_lines]
        assert any("good to hear from you" in t for t in texts)
        assert any("no hurry at all" in t for t in texts)

    def test_markdown_output_is_teleprompter_friendly(self, temp_dir):
        """The Markdown output should be readable as a teleprompter script."""
        source_file = temp_dir / "source.txt"
        source_file.write_text(
            "It was good to hear from you. "
            "Take all the time you need."
        )

        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        markdown_content = output.markdown_path.read_text()

        # Should contain style markers
        assert "This is the neutral reading" in markdown_content

        # Should contain utterance IDs
        assert "NEUTRAL-" in markdown_content

        # Should contain pause instructions
        assert "[Pause" in markdown_content

    def test_manifest_contains_required_fields(self, temp_dir):
        """The JSONL manifest should have stable utterance IDs and metadata."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("It was good to hear from you.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest_lines = output.manifest_path.read_text().strip().split("\n")

        assert len(manifest_lines) > 0
        first_entry = json.loads(manifest_lines[0])

        # Check required fields
        assert "id" in first_entry
        assert first_entry["id"].startswith("NEUTRAL-")
        assert "text" in first_entry
        assert "style" in first_entry
        assert "source" in first_entry
        assert first_entry["style"] == "neutral"

    def test_utterance_ids_are_stable(self, temp_dir):
        """Same source should generate same utterance IDs."""
        source_file = temp_dir / "source.txt"
        source_text = "It was good to hear from you."
        source_file.write_text(source_text)

        output1 = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest1 = output1.manifest_path.read_text()

        output2 = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest2 = output2.manifest_path.read_text()

        # IDs should be consistent across runs
        assert manifest1 == manifest2


class TestStyleHandling:
    """Test canonical style markers."""

    def test_default_style_is_neutral(self, temp_dir):
        """Default generated scripts use neutral style."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Some text.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest = output.manifest_path.read_text()
        first_entry = json.loads(manifest.strip().split("\n")[0])

        assert first_entry["style"] == "neutral"

    def test_accepts_multiple_styles_parameter(self, temp_dir):
        """Can generate script for multiple styles."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Good morning. How are you?")

        output = generate_script(
            source_files=[source_file],
            output_dir=temp_dir,
            styles=["neutral", "warm", "energetic"],
        )

        manifest_lines = output.manifest_path.read_text().strip().split("\n")
        styles_in_manifest = [json.loads(line)["style"] for line in manifest_lines]

        assert "neutral" in styles_in_manifest
        assert "warm" in styles_in_manifest
        assert "energetic" in styles_in_manifest

    def test_canonical_style_normalization(self, temp_dir):
        """If styles include 'netural', normalize to 'neutral'."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Test text.")

        output = generate_script(
            source_files=[source_file],
            output_dir=temp_dir,
            styles=["netural"],  # misspelling alias
        )

        manifest = output.manifest_path.read_text()
        first_entry = json.loads(manifest.strip().split("\n")[0])

        # Should normalize to canonical form
        assert first_entry["style"] == "neutral"


class TestProvenance:
    """Test source tracking and provenance."""

    def test_source_file_is_recorded_in_manifest(self, temp_dir):
        """Manifest should track which source file each utterance came from."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("It was good to hear from you.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest = output.manifest_path.read_text()
        first_entry = json.loads(manifest.strip().split("\n")[0])

        assert first_entry["source"] == str(source_file)

    def test_multiple_source_files_are_tracked(self, temp_dir):
        """Multiple source files should be distinguished in provenance."""
        source1 = temp_dir / "source1.txt"
        source1.write_text("First source text.")
        source2 = temp_dir / "source2.txt"
        source2.write_text("Second source text.")

        output = generate_script(
            source_files=[source1, source2], output_dir=temp_dir
        )
        manifest_lines = output.manifest_path.read_text().strip().split("\n")
        entries = [json.loads(line) for line in manifest_lines]

        sources = [entry["source"] for entry in entries]
        assert str(source1) in sources
        assert str(source2) in sources

        texts_by_source = {
            src: [e["text"] for e in entries if e["source"] == src]
            for src in (str(source1), str(source2))
        }
        assert texts_by_source[str(source1)] == ["First source text."]
        assert texts_by_source[str(source2)] == ["Second source text."]


class TestSessionAndBlockBoundaries:
    """Test script organization."""

    def test_manifest_includes_session_and_block_info(self, temp_dir):
        """Each utterance should indicate its session and block."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("First sentence. Second sentence.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        manifest = output.manifest_path.read_text()
        first_entry = json.loads(manifest.strip().split("\n")[0])

        assert "session" in first_entry
        assert "block" in first_entry

    def test_markdown_includes_session_markers(self, temp_dir):
        """Markdown should include visible session and block boundaries."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Text one. Text two. Text three.")

        output = generate_script(
            source_files=[source_file], output_dir=temp_dir, session_id=7
        )
        markdown = output.markdown_path.read_text()

        # The session must be stated explicitly, not inferred from ID prefixes
        assert "Session 7" in markdown


class TestFileHandling:
    """Test input/output file handling."""

    def test_accepts_pathlib_path(self, temp_dir):
        """Should accept pathlib.Path objects."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Test.")

        # Should not raise
        output = generate_script(source_files=[source_file], output_dir=temp_dir)
        assert output is not None

    def test_accepts_string_path(self, temp_dir):
        """Should accept string paths."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Test.")

        # Should not raise
        output = generate_script(
            source_files=[str(source_file)], output_dir=str(temp_dir)
        )
        assert output is not None

    def test_output_files_created_in_output_dir(self, temp_dir):
        """Output files should be created in specified directory."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Test.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        # Both files should be under temp_dir
        assert output.markdown_path.parent == temp_dir or temp_dir in output.markdown_path.parents
        assert output.manifest_path.parent == temp_dir or temp_dir in output.manifest_path.parents

    def test_error_on_missing_source_file(self, temp_dir):
        """Should raise error if source file doesn't exist."""
        missing_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            generate_script(source_files=[missing_file], output_dir=temp_dir)

    def test_error_on_unsupported_format(self, temp_dir):
        """Should raise error for unsupported file formats."""
        source_file = temp_dir / "source.unknown"
        source_file.write_text("Test.")

        with pytest.raises(ValueError):
            generate_script(source_files=[source_file], output_dir=temp_dir)


class TestCoverageMeasurement:
    """Test linguistic coverage reporting."""

    def test_output_includes_coverage_metrics(self, temp_dir):
        """ScriptOutput should include coverage metrics."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Good morning! How are you? I'm doing well.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        # Coverage metrics should be available
        assert hasattr(output, "coverage") or "coverage" in output.__dict__

    def test_coverage_metrics_have_concrete_values(self, temp_dir):
        """Coverage should report exact counts for a known input."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Good morning! How are you? I am 2 years old.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        assert output.coverage == {
            "unique_letters": 15,
            "total_utterances": 3,
            "total_words": 10,
            "has_questions": True,
            "has_exclamations": True,
            "has_numbers": True,
        }

    def test_coverage_reports_absent_features(self, temp_dir):
        """Absent punctuation and digits should read as False, not True."""
        source_file = temp_dir / "source.txt"
        source_file.write_text("Plain sentence one. Plain sentence two.")

        output = generate_script(source_files=[source_file], output_dir=temp_dir)

        assert output.coverage["has_questions"] is False
        assert output.coverage["has_exclamations"] is False
        assert output.coverage["has_numbers"] is False
        assert output.coverage["total_utterances"] == 2
