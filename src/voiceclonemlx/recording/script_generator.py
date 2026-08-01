"""Generate recording scripts from source material."""

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Set

from voiceclonemlx.ingestion.parsers import parse_source_file, split_into_sentences
from voiceclonemlx.recording.script_rendering import render_script
from voiceclonemlx.recording.script_storage import (
    write_manifest,
    write_script_markdown,
)
from voiceclonemlx.shared.styles import normalize_style, CANONICAL_STYLES


@dataclass
class Utterance:
    """A single utterance to be recorded."""

    id: str
    text: str
    style: str
    source: str
    session: int
    block: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ScriptOutput:
    """Output paths and metadata from script generation."""

    markdown_path: Path
    manifest_path: Path
    utterances: List[Utterance] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "markdown_path": str(self.markdown_path),
            "manifest_path": str(self.manifest_path),
            "utterance_count": len(self.utterances),
            "coverage": self.coverage,
        }


def generate_stable_id(style: str, text: str, source: str = "") -> str:
    """Generate a stable utterance ID derived from style, source, and text.

    The digest is computed from the style, the source file's *name* (not its
    directory, so moving a source does not renumber it), and the sentence
    text. Format: NEUTRAL-a1b2c3d4.

    Identical text repeated within one source and style produces the same ID;
    `generate_script` disambiguates those collisions with a suffix.
    """
    source_name = Path(source).name if source else ""
    payload = "\0".join([style, source_name, text]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{style.upper()}-{digest}"


def _disambiguate(utterance_id: str, seen: Set[str]) -> str:
    """Return a unique ID, suffixing repeats of identical text.

    Args:
        utterance_id: Content-derived ID, possibly already used
        seen: IDs already issued (mutated)
    """
    candidate = utterance_id
    repeat = 2
    while candidate in seen:
        candidate = f"{utterance_id}-{repeat}"
        repeat += 1

    seen.add(candidate)
    return candidate


def generate_script(
    source_files: List,
    output_dir,
    styles: Optional[List[str]] = None,
    session_id: int = 1,
) -> ScriptOutput:
    """Generate a recording script from source files.

    Args:
        source_files: List of source file paths (.txt supported)
        output_dir: Directory to write output files
        styles: List of styles to generate. Defaults to ["neutral"]
        session_id: Session number for this script

    Returns:
        ScriptOutput with paths to generated files and metadata

    Raises:
        FileNotFoundError: If source file doesn't exist
        ValueError: If file format unsupported or invalid style
    """
    # Normalize paths
    source_files = [Path(f) for f in source_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize styles
    if styles is None:
        styles = ["neutral"]
    styles = [normalize_style(s) for s in styles]

    # Parse all sources
    all_utterances = []
    seen_ids: Set[str] = set()

    for source_file in source_files:
        text = parse_source_file(source_file)
        sentences = split_into_sentences(text)

        # Create utterances for each style
        block_num = 1
        for style in styles:
            for sentence_idx, sentence in enumerate(sentences):
                if not sentence:
                    continue

                utterance_id = _disambiguate(
                    generate_stable_id(style, sentence, str(source_file)),
                    seen_ids,
                )
                utterance = Utterance(
                    id=utterance_id,
                    text=sentence,
                    style=style,
                    source=str(source_file),
                    session=session_id,
                    block=block_num,
                )
                all_utterances.append(utterance)

            block_num += 1

    # Generate Markdown output
    markdown_path = output_dir / f"script_session_{session_id}.md"
    markdown_content = render_script(all_utterances, styles, session_id)
    write_script_markdown(markdown_path, markdown_content)

    # Generate JSONL manifest
    manifest_path = output_dir / f"script_session_{session_id}.jsonl"
    write_manifest(manifest_path, all_utterances)

    # Calculate coverage
    coverage = _calculate_coverage(all_utterances)

    return ScriptOutput(
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        utterances=all_utterances,
        coverage=coverage,
    )


def _calculate_coverage(utterances: List[Utterance]) -> dict:
    """Calculate linguistic coverage metrics.

    Args:
        utterances: List of utterances

    Returns:
        Dictionary of coverage metrics
    """
    all_text = " ".join(u.text for u in utterances)

    # Extract letters (unique)
    letters = set(c.lower() for c in all_text if c.isalpha())

    # Count various features
    has_questions = any("?" in u.text for u in utterances)
    has_exclamations = any("!" in u.text for u in utterances)
    has_numbers = any(any(c.isdigit() for c in u.text) for u in utterances)

    return {
        "unique_letters": len(letters),
        "total_utterances": len(utterances),
        "total_words": len(all_text.split()),
        "has_questions": has_questions,
        "has_exclamations": has_exclamations,
        "has_numbers": has_numbers,
    }
