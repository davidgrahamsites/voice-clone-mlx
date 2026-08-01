"""Render utterances as a teleprompter-friendly Markdown script.

Pure rendering: this module takes utterances and returns a string. It never
reads or writes files. Utterances are duck-typed on `.id`, `.text`, and
`.style` so this module does not import the coordinator (no import cycle).
"""

from datetime import datetime
from typing import List


def escape_markdown(text: str) -> str:
    """Neutralize Markdown structure in untrusted source text.

    Source files are untrusted input: their text must read as words, never as
    headings, emphasis, code spans, links, or table cells. Line breaks are
    collapsed so a sentence cannot start a new Markdown block.

    Args:
        text: Raw sentence text from a source file

    Returns:
        Text safe to interpolate into a Markdown line
    """
    collapsed = " ".join(text.split())

    escaped = collapsed.replace("\\", "\\\\")
    for char in "`*_[]<>|~#":
        escaped = escaped.replace(char, "\\" + char)

    return escaped


def render_script(
    utterances: List, styles: List[str], session_id: int = 1
) -> str:
    """Render utterances as Markdown.

    Args:
        utterances: Utterances with `.id`, `.text`, and `.style`
        styles: Styles to emit sections for, in order
        session_id: Session number, stated explicitly in the header

    Returns:
        Markdown content
    """
    lines = []

    # Header
    lines.append("# Recording Script\n")
    lines.append(f"*Session {session_id}*\n")
    lines.append(
        f"*Generated: {datetime.now().isoformat()}*\n"
    )

    # Group utterances by style
    by_style = {}
    for u in utterances:
        if u.style not in by_style:
            by_style[u.style] = []
        by_style[u.style].append(u)

    # Generate blocks for each style
    for style in styles:
        if style not in by_style:
            continue

        lines.append(f"\n## {style.title()} Reading\n")
        lines.append(f"This is the {style} reading.\n")
        lines.append("[Pause for three seconds.]\n")

        for utterance in by_style[style]:
            lines.append(
                f"\n**{utterance.id}:** {escape_markdown(utterance.text)}\n"
            )
            lines.append("[Pause for two seconds.]\n")

    return "".join(lines)
