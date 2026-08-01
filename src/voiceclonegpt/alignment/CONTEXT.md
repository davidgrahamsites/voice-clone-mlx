# Alignment module

One job: turn provider-neutral diarization spans into a safe dataset admission
decision for one candidate clip.

## Inputs

- Candidate clip start/end offsets.
- Enrolled target speaker id.
- Diarization `SpeakerTurn` spans and overlap flags.

## Process

Ignore turns outside the candidate range. Reject the entire candidate when an
overlap flag is present, when another speaker intersects it, or when the target
speaker is absent. Accept only target-only, non-overlapping clips.

## Outputs

A `ClipDecision` with `accept` or `reject` plus a stable reason. This module
does not load models, edit audio, transcribe, or write manifests.

## Human check

Listen to accepted and rejected fixture clips and confirm that every mixed-
speaker clip is rejected before dataset construction.

## Style markers

`marker_parser.py` owns the **wording** of a spoken style marker — the one home
for it. It has two directions:

- `parse_style_marker(text) -> StyleMarker | None` — an utterance to a
  canonical style.
- `marker_phrase(style) -> str` — a canonical style to the exact sentence the
  reader is asked to say, and the teleprompter prints.

Both are pure: text in, text out. No audio, model, filesystem, network, or UI.
Standard library only.

### A marker is a whole utterance

`parse_style_marker` now anchors with `fullmatch`. It was `search`, which
matched a marker **buried inside ordinary speech** — "okay, this is the neutral
reading, let us begin" was accepted as a style boundary. That silently
relabels real speech as a marker, and the mislabelled audio then enters the
dataset. Tolerated: surrounding whitespace, case, an optional `the`/`a`,
trailing `.`/`!`/`?`, and repeated inner spaces. Other words are not.

### Only `netural` is aliased

That is the misspelling the reader actually produces. `nuetral`, `neutrl`,
`netrual` and friends return None rather than being guessed at — a fuzzy match
mislabels an entire recorded block, and a wrong style is worse than an
unrecognized one, because a human reviews the second and not the first.

### Behavior change

Rejecting embedded prose is a **narrowing** of what `parse_style_marker`
accepts. Nothing on `main` calls it outside its tests today (verified with
`git grep`), so no caller breaks — but if a future caller relies on finding a
marker inside a longer transcript segment, it must split into utterances first
rather than have this function loosened.

### TDD evidence

```bash
# red — the generator did not exist and embedded prose was accepted
$ python3 -m pytest tests/voice_studio/test_marker_parser.py -q
ERROR — ImportError: cannot import name 'marker_phrase'

# green — the original 4 tests are unchanged and still pass
$ python3 -m pytest tests/voice_studio/test_marker_parser.py -q
64 passed in 0.06s
$ python3 -m pytest -q
694 passed, 10 skipped in 5.94s
$ git diff --check
(clean)
```

`tests/voice_studio/test_marker_parser.py` (64) — phrase generation for all
eight styles, round-trip through the generated phrase, the `netural` alias in
both directions, case/article/punctuation/whitespace variants, embedded-prose
rejection, unknown styles, non-string input, and the frozen result.

## Marker windows

`marker_windows.py` turns timestamped transcript segments into the time span
each style governs. `split_style_windows(segments, recording_end=None)` reuses
the single marker parser; it does not decode audio or run Whisper.

Each marker is removed from the content. A window starts at the marker's end
and ends at the next marker's start, or at the supplied recording end, or at
the last transcript segment. Gaps are allowed, but overlapping transcript
segments are rejected because clipping would be ambiguous. A transcript with
no markers returns an explicit `no_markers` result. A marker with no content
after it is rejected rather than producing a zero-length training clip.

Inputs are copied and sorted without mutation. Times must be finite,
non-negative, and start before end; an explicit recording end is validated
even when the transcript is empty. Results are frozen dataclasses, and
malformed input raises `MarkerWindowError`.

TDD evidence:

    focused suite: 73 passed
    full suite with this seam: 820 passed, 9 skipped
    full suite with this seam and UI opt-in: 829 passed

## Whisper MLX invocation plan

`whisper_plan.py` builds, but never runs, the local command that a later
runner can use with the installed MLX Whisper package. It requires an existing
audio file, model path, and output directory. Remote-looking paths, missing
assets, directories in the wrong role, invalid language values, and non-path
inputs are refused before an argument list is returned.

The model path is always explicit, so a Hub identifier cannot silently trigger
a download. The plan uses the current interpreter (`sys.executable`) with
`-m mlx_whisper`, JSON output, the local model, output directory, optional
language, and the audio path. It imports no MLX package and starts no process;
a separate runner remains responsible for execution and review.

TDD evidence:

    focused suite: 62 passed
    baseline before this seam: 820 passed, 9 skipped
    after this seam: 882 passed, 9 skipped
    after this seam with UI opt-in: 891 passed

## Whisper JSON transcript parser

`whisper_json.py` is the pure handoff from a local MLX Whisper runner to
alignment. `parse_whisper_json(payload)` accepts a mapping, JSON text, or
UTF-8 bytes containing a `segments` list. Mapping implementations are
accepted through the standard mapping protocol; ordinary objects that merely
have a `segments` attribute are not payloads.

Each segment must have finite, non-negative numeric `start` and `end`
values with positive duration and non-blank string text. Text is preserved
exactly, including surrounding and repeated whitespace. Output is sorted into
an immutable `Transcript`; overlaps and duplicates are rejected, while gaps,
touching boundaries, and an empty list are valid. The caller's mapping and
list are never mutated. The seam performs no file, process, model, network, or
audio work.

TDD evidence (measured after integration):

    focused suite: 94 passed
    full suite: 976 passed, 9 skipped
    UI opt-in suite: 985 passed

## Alignment manifest rows

`alignment_rows.py` defines the pure, versioned `AlignmentManifest v1` row
contract used before audio can enter a dataset. Its separate
`alignment_row_schema.py` module owns field vocabulary and validation, while
the public names remain re-exported from `alignment_rows.py`.

`build_alignment_row` creates a pending row. `accept_row` and
`reject_row` perform explicit, attributed transitions and require an actor
and timestamp; the code cannot determine whether an actor is human, so the
human-review requirement remains a process rule around this artifact. Direct
construction is still possible for deserialization and tests, but frozen-row
invariants reject unknown states, unsigned decisions, and pending rows that
claim a reviewer.

`rows_to_jsonl` and `parse_alignment_jsonl` are deterministic and pure.
Parsing requires the exact v1 key set, revalidates every field, enforces
canonical styles and relative `master_audio` references, and rejects
unattributed or inconsistent decisions. No audio, model, clock, filesystem,
network, or process work occurs here.

TDD evidence (measured after integration):

    focused suite: 168 passed (98 + 70 across two modules)
    full suite: 1144 passed, 9 skipped
    UI opt-in suite: 1153 passed

## Script aligner

`script_aligner.py` pairs one validated style window with the expected
utterances for that style. `align_window(window, transcript,
expected_utterances, master_audio=...)` uses chronological order only: the
nth expected utterance pairs with the nth transcript segment inside the window.
It does not guess with fuzzy matching or reorder evidence.

Every result is a pending `AlignmentRow`. Mismatches remain reviewable through
the fixed reasons `expected_missing`, `observed_surplus`,
`text_mismatch`, and `outside_window`; `needs_review` is therefore a
pending row with reasons, never a fourth review state. Stored observed text is
verbatim, while comparison collapses whitespace and case-folds without
discarding punctuation. Surplus evidence is attached to the final expected row;
observations with no expected utterances are rejected because there is no row
to carry the evidence.

The seam reuses `StyleWindow`, `Transcript`, and `AlignmentRow` rather
than defining duplicates. It performs no audio, model, filesystem, clock,
network, or process work.

TDD evidence (measured after integration):

    focused command: 59 passed (44 + 15)
    full suite: 1203 passed, 9 skipped
    UI opt-in suite: 1212 passed
