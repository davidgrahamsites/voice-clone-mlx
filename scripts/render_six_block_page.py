"""Render the six-block recording page from the canonical prompt script."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/scripts/voice_training_script_30_minutes.md"
OUTPUT = ROOT / "docs/six-block-script.html"
PROMPT = re.compile(r"^(?P<id>[A-Z]+-[A-Z]-\d+):\s*(?P<text>.+)$")
MARKER = re.compile(r"^Say:\s+\*\*(?P<text>.+?)\*\*$")


def prompts() -> list[tuple[str, str, str | None]]:
    rows = []
    current_marker = None
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        marker = MARKER.match(line)
        if marker:
            current_marker = marker.group("text")
        match = PROMPT.match(line)
        if match:
            rows.append((match.group("id"), match.group("text"), current_marker))
    if len(rows) != 100:
        raise ValueError(f"expected 100 prompts, found {len(rows)}")
    return rows


def block_html(number: int, rows: list[tuple[str, str, str | None]]) -> str:
    start, end = rows[0][0], rows[-1][0]
    items = []
    previous_marker = None
    for prompt_id, text, marker in rows:
        if marker != previous_marker:
            items.append(f'          <li class="style-marker"><strong>Say: {html.escape(marker)}</strong><span class="pause">Pause three seconds.</span></li>')
            previous_marker = marker
        items.append(f'          <li><span class="prompt-id">{html.escape(prompt_id)}</span> {html.escape(text)}<span class="pause">Pause two seconds.</span></li>')
    items = "\n".join(items)
    return f"""      <section class=\"script-block\">
        <div class=\"script-heading\"><span class=\"block-number\">Block {number:02d}</span><span><strong>{start} → {end}</strong><small>Five-minute recording segment</small></span></div>
        <div class=\"script-body\"><p class=\"block-note\"><strong>Read the sentences only.</strong> Do not speak the prompt IDs. Pause silently for two seconds after each sentence.</p><ol>
{items}
        </ol></div>
      </section>"""


def render() -> str:
    rows = prompts()
    blocks = [rows[i * 100 // 6 : (i + 1) * 100 // 6] for i in range(6)]
    cards = "\n".join(block_html(i + 1, block) for i, block in enumerate(blocks))
    canonical = html.escape(SOURCE.read_text(encoding="utf-8"))
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#111210"><title>Six five-minute recording blocks — VoiceCloneMLX</title>
  <link rel="stylesheet" href="document.css">
</head>
<body>
  <header class="site-header"><div class="site-nav">
    <a class="wordmark" href="index.html">VOICECLONEMLX / RECORDING</a>
    <nav aria-label="Document navigation"><a href="index.html">Project</a><a href="six-block-script.html">Six blocks</a><a href="recording-script.html">Full script</a><a href="procedure.html">Procedure</a></nav>
  </div></header>
  <main class="doc-shell">
    <div class="doc-kicker">VoiceCloneMLX / complete six-block script</div>
    <h1>Five minutes at a time.</h1>
    <p class="block-note"><strong>This is the complete script.</strong> Open one block, record its sentences in order, and pause silently for two seconds after each one. The prompt IDs are visual labels; do not say them aloud.</p>
    <h2>Before every block</h2>
    <ol><li>Record 30 seconds of room tone with the AC unchanged.</li><li>Wait three seconds, read the calibration passage, then wait two seconds.</li><li>Read the open block's sentences only.</li><li>Stop after its final sentence and two-second pause; resume with the next block later.</li></ol>
    <section class="script-stack" aria-label="Complete recording script">
{cards}
    </section>
    <h2>After block six</h2>
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
