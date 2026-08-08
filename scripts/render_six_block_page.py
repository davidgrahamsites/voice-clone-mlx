"""Render the six short recording files from the canonical prompt script."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/scripts/voice_training_script_30_minutes.md"
OUTPUT = ROOT / "docs/six-block-script.html"
PROMPT = re.compile(r"^(?P<id>[A-Z]+(?:-[A-Z]+)?-\d+):\s*(?P<text>.+)$")
MARKER = re.compile(r"^Say:\s+\*\*(?P<text>.+?)\*\*$")
STYLE = re.compile(r"^## (?P<text>.+)$")
RECORDING_BLOCKS = (
    ("Neutral A opening", "neutral-a-opening", ("NEUTRAL-A-001", "NEUTRAL-A-028")),
    ("Neutral A finish and warm start", "neutral-a-warm", ("NEUTRAL-A-029", "WARM-006")),
    ("Warm finish, energetic, and serious", "warm-energetic-serious", ("WARM-007", "SERIOUS-009")),
    ("Neutral B opening", "neutral-b-opening", ("NEUTRAL-B-001", "NEUTRAL-B-028")),
    ("Neutral B finish and somber start", "neutral-b-somber", ("NEUTRAL-B-029", "SOMBER-006")),
    ("Somber finish, questions, emphasis, and dialogue", "somber-dialogue", ("SOMBER-007", "DIALOGUE-008")),
)


def prompts() -> list[tuple[str, str, str | None, str]]:
    rows = []
    current_marker = None
    current_style = None
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        style = STYLE.match(line)
        if style:
            current_style = style.group("text")
        marker = MARKER.match(line)
        if marker:
            current_marker = marker.group("text")
        match = PROMPT.match(line)
        if match:
            rows.append((match.group("id"), match.group("text"), current_marker, current_style))
    if len(rows) != 166:
        raise ValueError(f"expected 166 prompts, found {len(rows)}")
    return rows


def calibration() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    section = source.split("## Calibration passage", 1)[1].split("---", 1)[0]
    passage = section.split("At first light,", 1)[1]
    return "At first light, " + " ".join(line.strip() for line in passage.splitlines() if line.strip())


def block_html(number: int, title: str, slug: str, rows: list[tuple[str, str, str | None, str]], calibration_text: str) -> str:
    start, end = rows[0][0], rows[-1][0]
    filename = f"block-{number:02d}-{slug}.wav"
    items = []
    previous_marker = None
    for prompt_id, text, marker, _style in rows:
        if marker != previous_marker:
            items.append(f'          <li class="style-marker"><strong>Say: {html.escape(marker)}</strong><span class="pause">Pause three seconds.</span></li>')
            previous_marker = marker
        items.append(f'          <li><span class="prompt-id">{html.escape(prompt_id)}</span> {html.escape(text)}<span class="pause">Pause two seconds.</span></li>')
    items = "\n".join(items)
    return f"""      <section class=\"script-block\">
        <div class=\"script-heading\"><span class=\"block-number\">File {number:02d}</span><span><strong>{html.escape(title)}</strong><small>{start} to {end}</small></span></div>
        <div class=\"script-body\"><p class=\"block-note\"><strong>Start and finish with calibration.</strong> Read the passage below without announcing it, then pause two seconds before the first prompt.</p><p class=\"calibration\">{html.escape(calibration_text)}</p><p class=\"block-note\"><strong>Read the sentences only.</strong> Do not speak the prompt IDs. Pause silently for two seconds after each sentence.</p><ol>
{items}
        </ol><p class="block-file">Save this recording as: {filename}</p></div>
      </section>"""


def render() -> str:
    rows = prompts()
    calibration_text = calibration()
    cards = []
    row_by_id = {row[0]: index for index, row in enumerate(rows)}
    for number, (title, slug, (start_id, end_id)) in enumerate(RECORDING_BLOCKS, start=1):
        start_index = row_by_id[start_id]
        end_index = row_by_id[end_id]
        group = rows[start_index : end_index + 1]
        cards.append(block_html(number, title, slug, group, calibration_text))
    cards = "\n".join(cards)
    canonical = html.escape(SOURCE.read_text(encoding="utf-8"))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#111210"><title>Six five-minute recording files - VoiceCloneMLX</title>
  <link rel="stylesheet" href="document.css">
</head>
<body>
  <header class="site-header"><div class="site-nav">
    <a class="wordmark" href="index.html">VOICECLONEMLX / RECORDING</a>
    <nav aria-label="Document navigation"><a href="index.html">Project</a><a href="six-block-script.html">Six files</a><a href="recording-script.html">Full script</a><a href="procedure.html">Procedure</a></nav>
  </div></header>
  <main class="doc-shell">
    <div class="doc-kicker">VoiceCloneMLX / six five-minute files</div>
    <h1>Six short files, expressive voice.</h1>
    <p class="block-note"><strong>This is the complete expressive script.</strong> Record one file at a time. Each file contains 26 to 28 prompts, so neutral is spread across the session instead of becoming one long recording. The prompt IDs are visual labels; do not say them aloud.</p>
    <h2>Before every file</h2>
    <ol><li>Record 30 seconds of room tone with the AC unchanged.</li><li>Wait three seconds, read the calibration passage, then wait two seconds.</li><li>Read the open block's sentences only.</li><li>Stop after its final sentence and two-second pause; resume with the next block later.</li></ol>
    <section class="script-stack" aria-label="Complete recording script">
{cards}
    </section>
    <h2>After the final file</h2>
    <p>Record the closing calibration and room tone. Keep every file, including rejected takes, until the review record is complete.</p>
    <section class="canonical-source"><h2>Canonical script — verbatim source</h2><pre>{canonical}</pre></section>
    <p><a class="source-link" href="recording-script.html">Open the master script ↗</a> <a class="source-link" href="../data/scripts/voice_training_script_30_minutes.md">Open the Markdown source ↗</a></p>
  </main>
  <footer class="site-footer"><div class="site-footer-inner"><span>Private, local-first, and explicit about what has been verified.</span><a href="index.html">Back to the project page →</a></div></footer>
</body></html>
'''


if __name__ == "__main__":
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"rendered {OUTPUT}")
