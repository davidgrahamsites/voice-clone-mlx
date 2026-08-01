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

### Malformed spans are refused, not worked around

`decide_clip` and `SpeakerTurn` now raise `OverlapGateError` (a `ValueError`)
for non-numeric, `bool`, non-finite, negative, or non-advancing times; blank or
non-string speaker ids; a non-`bool` overlap flag; and a `turns` argument that
is not a sequence of `SpeakerTurn`. The public signature of `decide_clip` is
unchanged and every valid input decides exactly as before — the four original
test bodies are byte-identical and still pass; only the import block grew.

`ClipDecision` also gained an `accepted` property and the `ACCEPT_STATUS` /
`REJECT_STATUS` constants, so a caller asking "was this accepted?" never spells
the status string itself. That string now has one home.

The motivating case is `NaN`: it compares False against everything, so a turn
with a `NaN` end would fail the intersection test and a guest talking over the
clip would look like it was outside it. The gate would then accept mixed audio.
A safety decision made from a malformed span is worse than no decision, so
there is no coercion or default here.

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

The audio check is public as `resolve_local_audio(audio_path) -> Path` so other
seams can ask the same question without a second copy of the rule; the builder
itself now calls it. See the Free Speech Mode planner section below.

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

## Whisper runtime runner

`whisper_runner.py` is the only runtime seam for one existing `WhisperPlan`.
`run_whisper_plan(plan, master_audio=, transcriber_version=, executor=,
output_reader=, timeout_s=)` passes that plan's argv unchanged to the local
process exactly once. The plan already fixes the backend as the current
interpreter running `-m mlx_whisper` with an explicit local model; the runner
does not import MLX or discover/download a model.

The default reader accepts only the JSON filename the local command-line tool
writes: `<audio stem>.json` in the already-validated output directory. Its
payload goes through `parse_whisper_json`, then the existing transcription
manifest builder and serializer. The returned JSONL preserves the caller's
safe relative `master_audio`, uses the plan language, and records
`mlx_whisper` plus caller-supplied version. No manifest schema changes.

Execution is bounded by a positive finite timeout (120 seconds by default),
has no retry or fallback, and discards process output rather than accumulating
unbounded logs. Failures from execution, reading, parsing, or manifest
construction become `WhisperRunnerError`. Tests inject both executor and
output reader, so they run no process, model, or audio.

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

## Transcription manifest

`transcription_manifest.py` records an already-parsed `Transcript` as durable
JSONL, giving the chain a reviewable artifact between "Whisper JSON was parsed"
and "windows and alignment are computed". `transcription_row_schema.py` holds
the vocabulary — schema version, key set, per-field validators — and
`transcription_manifest.py` re-exports every public name, so callers import one
module. Provider-neutral: `transcriber` and `transcriber_version` are
caller-supplied strings recorded verbatim and never detected; nothing here
knows what MLX, Whisper, or Qwen are.

Segment ids are positional — `<master_audio stem>-<index:04d>` — so the same
transcript always yields the same ids. Because they are derived, the parser
recomputes them and refuses a manifest whose rows were renamed, reordered, or
dropped. `language` is optional, caller-supplied, and omitted from JSON when
unset so its absence is a fact rather than a null.

Every field rule lives in `__post_init__`, so direct construction and parsing
face identical checks: a rule only the builder enforced would be bypassed by a
hand-edited manifest. Refused are absolute, traversing, or schemed
`master_audio`; ids containing path separators; times that are non-finite,
negative, non-advancing, or `bool` (an `int` subclass, so `True` must not pass
as 1); blank text; a foreign `schema_version`; and any unknown or missing key.
Text is stored byte for byte — Whisper's spacing is evidence.

Ordering and non-overlap are not restated here. `parse_transcription_jsonl`
rebuilds a payload and runs `parse_whisper_json`, so that rule keeps its one
home; a written artifact is refused rather than silently reordered. The seam
opens no file, starts no process, and imports no backend, network, or clock.

TDD evidence (measured at this worktree):

    focused: PYTHONPATH=src python3 -m pytest \
      tests/voice_studio/test_transcription_manifest.py \
      tests/voice_studio/test_transcription_manifest_safety.py -q
      135 passed (49 + 86)
    baseline (both ignored): 1580 passed, 9 skipped
    full suite:              1715 passed, 9 skipped

`PYTHONPATH=src` is required: this worktree has no `conftest.py`, `setup.py`,
or installed package, so no test collects without it. Pre-existing, unrelated
to this seam.

## Free Speech Mode candidate planner

`free_speech_plan.py` is the seam between diarization and transcription for
recordings that have no script. `plan_free_speech(candidates, target_speaker=,
model_path=, output_dir=, language=None)` returns one frozen
`PlannedCandidate` per input `FreeSpeechCandidate`, in input order.

It owns no policy. Admission is `overlap_gate.decide_clip` and the command is
`whisper_plan.build_whisper_plan`; both are called through module-level names,
and the safety tests monkeypatch them to prove the delegation rather than
assert on a re-implementation. The gate's reason vocabulary appears nowhere in
this module's code literals.

### Nothing is dropped

A rejected candidate stays in the result with `whisper_plan=None` and its
`ClipDecision`. A planner that returned only accepted rows would make "the gate
refused this" indistinguishable from "the diarizer never proposed it", and only
the first is worth a reviewer's attention.

### Nothing is salvaged

Rejection is whole-candidate. The clean-looking opening seconds of a clip a
guest talks over are not kept, and an accepted candidate is planned at its full
span. A diarizer boundary is an estimate; trimming to it trades a reviewed
rejection for an unreviewed guess about where the second voice starts. There is
no parameter, result field, or identifier in the module that would enable it,
and a test asserts that by AST.

### Every candidate's audio is validated, accepted or not

`whisper_plan.resolve_local_audio(audio_path) -> Path` is the one home for
"is this an existing local audio file?". `build_whisper_plan` calls it, and so
does the planner — for **every** candidate, before any gate decision, in a
single pass over the batch. A remote or missing path refuses the whole call.

Validation is deliberately not skippable by being rejected. A bad path is a
fault in whatever produced the batch, and surfacing it only for the candidates
the gate happened to accept would make the error depend on who was talking in
the room. Rejected candidates still carry their resolved absolute
`audio_path`; what they do not get is a Whisper plan.

### Cap

At most `MAX_CANDIDATES` candidates per call — the module is the one home for
that number, so it is not restated here. The length is measured **before** the
batch is materialized, so an oversized lazily-loaded sequence is refused
without being read; a test double raising on `__iter__`/`__getitem__` proves
it. One over the cap raises `FreeSpeechPlanError` before the gate is consulted
even once. An unbounded
diarizer result becomes an explicit error rather than a silently enormous batch
of work that no one has time to review.

Inputs are never mutated and results are frozen dataclasses and tuples. The
module imports no process, network, audio, or model package, starts nothing,
and writes nothing — **no transcription is executed here**. It produces an argv
a separate runner, which does not exist yet, would have to run.

### Human check

For a recording with a known guest interruption, confirm every candidate
touching the interruption carries `whisper_plan=None`, and that no planned span
is shorter than the candidate the diarizer proposed.

### TDD evidence (measured at this worktree)

    red — before implementation:
      ModuleNotFoundError: No module named
        'voiceclonegpt.alignment.free_speech_plan'
      ImportError: cannot import name 'OverlapGateError'
      (3 errors during collection)

    focused: PYTHONPATH=src python3 -m pytest \
      tests/voice_studio/test_overlap_gate.py \
      tests/voice_studio/test_free_speech_plan.py \
      tests/voice_studio/test_free_speech_plan_safety.py \
      tests/voice_studio/test_whisper_plan.py -q
      163 passed

    baseline:   2009 passed, 9 skipped
    full suite: 2106 passed, 9 skipped (+97)
