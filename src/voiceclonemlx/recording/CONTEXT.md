# Recording — script generation and local capture

Turns source material into a recording script: a teleprompter Markdown file to
read from, and a JSONL manifest that is the durable record of what was said,
in what style, from which source.

## Inputs

- **Source files** — paths to `.txt` or `.docx` files. Untrusted content.
  Parsing and format limits belong to `voiceclonemlx.ingestion.parsers`, not
  here.
- **Styles** — optional list of style names; defaults to `["neutral"]`.
  Normalized by `voiceclonemlx.shared.styles.normalize_style`.
- **Output directory** — created if missing.
- **Session id** — integer, defaults to `1`. Names the output files and is
  stated in the Markdown header.

## Process

`script_generator.generate_script` is the coordinator. It parses, constructs
utterances, sequences them, then calls the two seams:

| Responsibility | Home |
|---|---|
| Parse sources into sentences | `ingestion.parsers` (not this folder) |
| Normalize styles | `shared.styles` (not this folder) |
| Build + sequence utterances, assign IDs | `script_generator.py` |
| Render Markdown, escape untrusted text | `script_rendering.py` |
| Write the JSONL manifest | `script_storage.py` |
| Coverage metrics | `script_generator._calculate_coverage` (pure helper) |

Utterance IDs come from `generate_stable_id`: `STYLE-<sha256[:8]>` over the
style, the source file's *name*, and the sentence text. Moving a source file
does not change its IDs; editing a sentence changes only that sentence's ID.
Identical repeated text gets a `-2`/`-3` suffix from `_disambiguate`.

The renderer escapes source text (`escape_markdown`) so a source file cannot
inject headings, emphasis, code spans, links, or table rows into the script.
The manifest deliberately stores text **raw** — escaping is a presentation
concern, not a record of what was said.

Neither `script_rendering` nor `script_storage` imports `script_generator`;
they duck-type utterances on `.id`/`.text`/`.style` and `.to_dict()`. That
keeps the cycle out and makes both seams testable with hand-built objects.

`capture.capture_session` accepts one injected capture callable and persists
its provider-neutral signed pulse-code modulation (PCM) frames as a lossless
WAV plus an append-only JSON record. `macos_microphone.MacOSMicrophoneCapture`
is the optional macOS provider for that callable. It uses Audio Queue Services
through the Python standard library and has no third-party dependency.

Importing or constructing the provider does not load the native framework,
open an input device, or ask for permission. Calling it is the sole native
boundary: Audio Queue Services opens the input then, so macOS can show its
permission prompt only after the Studio user explicitly chooses Record. A
take is bounded to 60 seconds and defaults to five seconds of mono, 24 kHz,
signed 16-bit PCM. Deleting `macos_microphone.py` leaves `capture.py` and every
provider-neutral recording artifact unchanged.

## Outputs

Written to the output directory:

- `script_session_{n}.md` — teleprompter script: `# Recording Script`, a
  `*Session {n}*` line, then one `## <Style> Reading` section per style, each
  utterance as `**<ID>:** <escaped text>` with pause cues.
- `script_session_{n}.jsonl` — one JSON object per utterance with `id`, `text`,
  `style`, `source`, `session`, `block`. UTF-8, non-ASCII preserved.

Returned in memory: `ScriptOutput` with both paths, the utterance list, and a
coverage dict (`unique_letters`, `total_utterances`, `total_words`,
`has_questions`, `has_exclamations`, `has_numbers`).

Local capture returns `CapturedPcm`; `capture_session` writes `<take>.wav` and
`<take>.json` into an existing caller-selected directory.

## Human check

Before recording a session:

1. Read the Markdown aloud — it should be readable straight through. Escaping
   leaves backslashes visible in the raw file; if the source was full of
   punctuation, confirm it still reads cleanly in a Markdown viewer.
2. Confirm the `*Session n*` header matches the session you intend to record.
3. Spot-check that manifest IDs match the Markdown IDs for a few utterances.
4. Check the coverage dict is not lopsided — no questions or no exclamations
   across a whole session means the source material is too uniform to train on.
5. In the packaged Voice Studio app, choose Record and confirm macOS asks for
   microphone access then—not when the app opens—and that denial is reported
   in the status bar without writing a take.

## Tests

`tests/voice_studio/test_script_generation_core.py` covers the coordinator and
domain behavior. `tests/voice_studio/test_script_generation_seams.py` covers
rendering, storage, ID stability, and Markdown injection. Red/green evidence
for the current design is recorded in
`docs/verification/script-generator-tdd.md`.
`tests/voice_studio/test_recording_capture.py` covers persistence with fake PCM;
`tests/voice_studio/test_macos_microphone.py` covers the optional provider with
a fake native boundary and never opens a microphone.
