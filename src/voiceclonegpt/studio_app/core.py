"""Voice Studio app logic, free of any UI toolkit so it can be tested headlessly."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from voiceclonegpt.ingestion.parsers import parse_source_file, split_into_sentences
from voiceclonegpt.recording.script_generator import ScriptOutput, generate_script
from voiceclonegpt.shared.integration_seam import emit_app_event
from voiceclonegpt.shared.styles import CANONICAL_STYLES

from . import APP_NAME


@dataclass
class LoadedScript:
    """A source file parsed into the lines that would be recorded."""

    path: Path
    lines: List[str]


class StudioSession:
    """Holds what the Studio window is currently working on."""

    def __init__(self) -> None:
        self.script: Optional[LoadedScript] = None
        self.last_output: Optional[ScriptOutput] = None

    @property
    def styles(self) -> List[str]:
        return sorted(CANONICAL_STYLES)

    def load_script(self, path) -> LoadedScript:
        """Parse a source file into normalized lines.

        Raises FileNotFoundError / ValueError from the ingestion layer unchanged.
        """
        path = Path(path)
        lines = split_into_sentences(parse_source_file(path))
        self.script = LoadedScript(path=path, lines=lines)
        emit_app_event(
            APP_NAME, "script_loaded", {"source": str(path), "line_count": len(lines)}
        )
        return self.script

    def generate_recording_script(
        self, output_dir, styles: Optional[List[str]] = None, session_id: int = 1
    ) -> ScriptOutput:
        """Run the existing generator over the loaded source file."""
        if self.script is None:
            raise ValueError("Load a script file first.")
        output = generate_script(
            source_files=[self.script.path],
            output_dir=output_dir,
            styles=styles,
            session_id=session_id,
        )
        self.last_output = output
        emit_app_event(
            APP_NAME,
            "script_generated",
            {"manifest": str(output.manifest_path), "coverage": output.coverage},
        )
        return output

    def record_line(self, index: int) -> str:
        """Placeholder for recording. No capture backend exists yet.

        Returns a message for the UI to display rather than raising, so the
        button is safe to press.
        """
        if self.script is None:
            return "Load a script file first."
        if not 0 <= index < len(self.script.lines):
            return "Select a line to record."
        emit_app_event(APP_NAME, "record_requested", {"line_index": index})
        return f"Recording is not implemented yet (line {index + 1})."

