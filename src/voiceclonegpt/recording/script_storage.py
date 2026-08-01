"""Write recording-script manifests to disk.

The only writer of the JSONL manifest. Utterances are duck-typed on
`.to_dict()` so this module does not import the coordinator (no import cycle).
"""

import json
from pathlib import Path
from typing import List


def write_script_markdown(script_path: Path, content: str) -> None:
    """Write the rendered Markdown script, replacing any existing file.

    Args:
        script_path: Path to write the script to
        content: Rendered Markdown from `script_rendering.render_script`
    """
    Path(script_path).write_text(content, encoding="utf-8")


def write_manifest(manifest_path: Path, utterances: List) -> None:
    """Write utterances as a JSONL manifest, replacing any existing file.

    Text is stored raw (unescaped) — Markdown escaping belongs to the
    renderer, not to the record of what was said.

    Args:
        manifest_path: Path to write the manifest to
        utterances: Utterances exposing `.to_dict()`
    """
    with open(manifest_path, "w", encoding="utf-8") as f:
        for utterance in utterances:
            line = json.dumps(utterance.to_dict(), ensure_ascii=False)
            f.write(line + "\n")
