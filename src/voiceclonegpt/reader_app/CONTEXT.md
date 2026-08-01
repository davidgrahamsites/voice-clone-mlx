# reader_app — Voice Reader

**Job:** the launchable app shell for reviewing recorded takes.

**Reads:** a JSONL recording manifest (`script_session_N.jsonl` as written by
`recording.script_generator`) and audio files named `<utterance id>.<ext>`
where ext is one of `.wav`, `.m4a`, `.mp3`, `.aiff`. Audio is looked up in the
manifest's own directory unless another directory is given.

**A manifest is untrusted input.** It may have been hand-edited, copied from
elsewhere, or produced by another tool, so it is validated rather than trusted:

- Every row must be a JSON **object** with `id`, `text`, and `style` all
  present and all strings. Anything else raises `ValueError` naming the file,
  the line number, and the offending field.
- An `id` must be usable as a plain file name. Empty/blank ids, ids containing
  `/`, `\`, or NUL, absolute paths, and `.`/`..` are rejected
  (`is_safe_utterance_id`).
- `find_audio` resolves both the candidate and `audio_dir` and returns a path
  only when the resolved candidate is **inside** the resolved `audio_dir` and
  is a real file. A symlink pointing outside is refused; a symlink inside is
  fine; a directory named `X.wav` is not audio.
- It returns the **resolved** path — the exact path that was checked. Returning
  the unresolved candidate would mean the caller acts on a path that could
  resolve somewhere else than the one that passed containment. For a symlink
  inside the directory this means callers see the link's target, not the link.

The rule is containment, not just string cleaning: an id can never make the
Reader open a file outside the directory the user chose.
**Does:** lists each take and plays the ones that have audio.
**Writes:** nothing to disk.
**Human check:** `PYTHONPATH=src python3 -m voiceclonegpt.reader_app` opens a
window; opening a manifest lists its takes, marking which have audio.

## Files

- `core.py` — `Take`, `load_manifest`, `is_safe_utterance_id`, `find_audio`,
  `Player`. No UI toolkit.
- `synthesis_session.py` — `ReaderSynthesisSession`, `is_safe_output_name`.
  Generates WAV or MP3 takes through the shared round-trip contract. No UI
  toolkit.
- `audio_export.py` — output-format validation and local WAV-to-MP3 encoding.
  WAV passes through unchanged; MP3 uses an installed local encoder and never
  downloads a codec.
- `synthesis_controller.py` — `ReaderSynthesisController`, `VoiceChoice`.
  Coordinates verified bundle selection, runtime readiness, text input, and
  synthesis requests. No UI toolkit and no backend imports.
- `manifest_roundtrip.py` — `generate_missing_takes`. Fills in a script
  manifest's missing audio, beside the manifest. No UI toolkit.
- `ui.py` — `ReaderWindow` (Tk widgets only) plus `build_app()` / `main()`.
- `__main__.py` — `python3 -m voiceclonegpt.reader_app`.

## Seams

- `Player(command=...)` — the playback binary, `afplay` by default.
- `ReaderWindow(master, player=..., choose_file=...)` — inject a player and a
  file-picker stub; constructing the window never starts an event loop.
- Bus integration attaches at `shared.integration_seam.set_event_sink`. This
  folder emits `manifest_loaded` and `playback_started`, and never imports
  `voiceclonegpt.bus` directly.
- `ReaderSynthesisSession(..., round_trip=...)` — inject the round-trip
  callable. The default lazily imports
  `voiceclonegpt.shared.roundtrip.run_round_trip`.
- `ReaderSynthesisController(registry=..., verified_runtime_ids=...,
  bundle_reader=..., readiness=..., session_factory=...)` — inject every
  provider-facing boundary. Production passes no verified real runtime ids;
  tests use fakes without opening a model.

## Gated cloned-voice synthesis

`ReaderSynthesisController.inspect_bundle(bundle_dir) -> VoiceChoice` verifies
the bundle checksum before showing its identity and declared runtime ids.
`select(bundle_dir, runtime_id) -> ReadinessReport` clears any older selection,
then requires that the runtime is declared by that verified manifest, is not
the silent `null` placeholder, is explicitly human-verified, is registered,
and passes every no-load readiness check. Any failed or incomplete check leaves
`ready` false.

Text may be entered with `set_text(text)` or read exactly as UTF-8 with
`ingest_text(path)`. `synthesize(output_dir, output_name) -> Path` delegates to
`ReaderSynthesisSession`, preserving its confined atomic-write rules. The Tk
window keeps Generate disabled until both a voice is ready and text is
nonblank. Choosing another bundle or runtime invalidates that UI state.

The production composition root still registers only `null` and passes an
empty `verified_runtime_ids` set. `MlxQwenRuntime` is therefore not selectable;
registration and the verification allow-list must change together only after
a human listens to real local output.

## Synthesis session

`ReaderSynthesisSession(bundle_dir, runtime_id, runtime, output_dir)` turns
text into a WAV or MP3 take on disk. `synthesize(text, output_name) -> Path`;
the filename extension must be `.wav` or `.mp3`.

It owns **only** request validation, the call, local format conversion, and the
write. Bundle
verification and runtime dispatch belong to `shared.roundtrip`; model loading
to `synthesis/`; choosing a bundle or runtime to the caller. It never selects,
downloads, or converts anything.

**`output_name` is a file name, never a path.** Absolute names, separators,
`.`/`..`, and NUL are refused, and the resolved target must sit directly in the
resolved output directory. A **pre-placed symlink at the target is refused
rather than followed** — otherwise a link planted in the output directory would
redirect the write anywhere on disk. Verified against a real symlink: the
target file is left byte-identical.

**Writes are atomic enough for local use, and the temp file is created
exclusively.** Bytes go to a sibling created by `tempfile.mkstemp`
(`O_CREAT | O_EXCL`, unguessable name), then `os.replace` moves it into place.
A reader sees either the previous take or the new one, never a half-written
file; the sibling keeps the replace on one filesystem. If the replace fails,
the temp file is removed and the previous take survives. This is not a
durability guarantee across power loss.

The exclusive creation is load-bearing, not tidiness. An earlier version wrote
to a predictable `<target>.partial` with `Path.write_bytes`, which **follows a
symlink**: anyone able to create a file in the output directory could plant
`a.wav.partial -> /somewhere/important` and have the next take overwrite it.
`mkstemp` refuses to open anything that already exists, so a planted link is
simply ignored.

Note the side effect: `mkstemp` creates at mode `0600`, so takes are
owner-readable only (previously `0644`). For personal voice recordings that is
the better default, but it is a change — if a take needs to be shared, widen it
deliberately rather than loosening this.

### TDD evidence for the temp-file fix

```bash
# red — the planted symlink was followed
$ python3 -m pytest tests/voice_reader/test_synthesis_session.py -q
FAILED …::TestAtomicWrite::test_a_planted_partial_symlink_cannot_redirect_the_write
FAILED …::TestAtomicWrite::test_the_temp_path_is_not_the_predictable_partial_name
2 failed, 39 passed in 0.25s

# green — after switching to tempfile.mkstemp
$ python3 -m pytest tests/voice_reader/test_synthesis_session.py -q
41 passed in 0.26s
$ python3 -m pytest tests/voice_reader/ -q
198 passed, 5 skipped in 0.49s
$ python3 -m pytest -q
488 passed, 9 skipped in 5.21s
$ git diff --check
(clean)
```

Verified outside the suite too: with `out/a.wav.partial` symlinked to a file
holding `ORIGINAL`, the take is written correctly and the victim file is still
`ORIGINAL`. The planted symlink itself is left in place — this module deletes
only files it created.

**One typed error.** Empty text, unsafe name, missing output directory,
round-trip failure, non-`bytes` audio, and write failure all raise
`ReaderSynthesisError`, with the provider exception chained on `__cause__`.
Nothing is written when synthesis fails.

### Blocked on a branch, and honest about it

`voiceclonegpt.shared.roundtrip` **does not exist in this checkout** — it lives
on `fix/v0.6.2-model-roundtrip-contract`, and `shared/` was outside the write
scope for this change. So the default round trip imports it lazily and raises a
clear `ImportError` naming the module. Every rule above is fully tested through
an injected callable; **the wiring to the real contract is not yet exercised**
and must be re-run once the branches meet.

## Boundaries

Must not import `studio_app`. May import `shared/`. Deleting this folder must
leave every other module working.

## Tests and evidence

Two modules, split by responsibility:

| File | Covers | Needs a display? |
|---|---|---|
| `tests/voice_reader/test_reader_manifest_security.py` | manifest loading, row schema, unsafe ids, traversal and symlink confinement | no — pure logic and temp files, runs in ~0.1 s |
| `tests/voice_reader/test_reader_app.py` | `Player`, `ReaderWindow`, and the headless-display gate | only the window tests, and only under the opt-in below |
| `tests/voice_reader/test_manifest_roundtrip.py` | 37 tests: generation and ordering, resume across every accepted format, failure and re-run, input validation, and that no borrowed rule is re-implemented | no |
| `tests/voice_reader/test_manifest_roundtrip_writer.py` | 13 tests: placement beside the manifest, and that the writer cannot be bypassed | no |
| `tests/voice_reader/test_synthesis_session.py` | 41 tests: the round-trip call, input validation, output confinement, typed failures, atomic replacement, and the lazy round-trip seam | no |
| `tests/voice_reader/test_synthesis_controller.py` | verified-before-load ordering, declared/non-null/human-verified/registered/readiness gates, exact text/runtime/bundle flow, text ingestion, and non-silent fake WAV output | no |

The security module deliberately imports no Tk and reads no environment
variable: the rules that keep an untrusted manifest from reaching outside the
audio directory must be verifiable on any machine, with no opt-in and no
display. The `manifest` fixture is defined in both files — a shared home would
need a support module, which is outside the scope this split was granted.

`TestManifestSchema` and `TestFindAudioIsConfined` (now in the security
module) were written first and failed against the previous implementation —
**17 failed, 41 passed** — because rows were read with
`record.get(field, default)` and `find_audio` returned any path that merely
`exists()`. A manifest row of `{"id": "../../etc/passwd", ...}` produced a
`Take` pointing outside the audio directory. After the fix `find_audio` returns
the resolved, containment-checked path.

Current runs: **314 passed, 9 skipped** for the default full suite (the 9 are
the real-window tests), and **323 passed, 0 skipped** with
`VOICECLONEGPT_RUN_UI_TESTS=1`. Splitting the 480-line test module into 291 +
218 lines changed no test id in either direction — verified by diffing the
collected ids before and after.

One pre-existing test changed: `test_malformed_line_reports_its_line_number`
wrote `{"id": "A"}` on line 1, which is now itself invalid, so the error came
from line 1 instead of the malformed line 2 it was checking. Line 1 is now a
complete valid row; the assertion is unchanged.

### Headless display seam

Tk tests use `reader_tk_root`, defined in this app's own test module. It skips
**only** on a recognized display failure and re-raises any other `TclError`, so
a real widget bug cannot hide as an environment skip.

`tkinter.Tk` is constructed in exactly one place, `_open_hidden_root`, behind an
**explicit opt-in**:

```bash
python3 -m pytest tests/voice_reader/test_reader_app.py                       # UI tests skipped
VOICECLONEGPT_RUN_UI_TESTS=1 python3 -m pytest tests/voice_reader/test_reader_app.py   # UI tests run
```

Without `VOICECLONEGPT_RUN_UI_TESTS=1` the real-window tests skip **before Tk is
initialized**. This is not caution about noisy failures: in a headless runner Tk
can *abort the process*, and an abort cannot be caught by any `except` clause,
so the earlier try/except seam still killed the run. Only the value `1` opts in;
`true`, `yes`, and `0` do not.

The opt-in does not weaken the tests. With it set, real windows are built and a
widget bug fails the run — `test_opt_in_lets_construction_proceed` asserts an
exception raised during construction propagates rather than becoming a skip.

`factory` and `env` are injectable, so every branch is tested on a machine that
has a display: `TestHeadlessDisplaySeam` covers the gate (absent opt-in skips,
non-`1` values skip, Tk is never constructed without opt-in), five known
no-display messages, an unrelated `TclError`, a non-Tcl exception, and a
meta-test that no test constructs a root outside the helper.

The name is app-specific for two reasons:

1. **Independent deletability.** This folder's contract says deleting it must
   leave everything else working — and the reverse. A fixture shared with Voice
   Studio would make Reader's tests fail when Studio is removed. Each app owns
   its display seam so neither can break the other.
2. **No silent fallback.** An earlier version named it `tk_root`, which shadowed
   the permissive fixture in `tests/conftest.py`. Under that name, deleting or
   misspelling the local fixture would have quietly fallen through to the
   conftest one, which skips on *any* `TclError` — turning a real widget failure
   into a green "skipped" run. With an app-specific name a missing fixture is a
   collection error instead.

Voice Studio has the mirror-image fixture, `studio_tk_root`. The two bodies are
similar by nature — both open a hidden Tk root — but they are separate facts
about separate apps, not one fact stored twice.

## Manifest round trip

`generate_missing_takes(manifest_path, synthesize) -> RoundTripResult` walks a
JSONL script manifest and synthesizes every utterance that has no audio yet.
There is no `writer` argument — see below.

**Reads:** the manifest. **Writes:** `<utterance_id>.wav` into the manifest's
own directory. **Does:** orchestration, and nothing else.

### Every rule it depends on already has a home

| Concern | Owner |
|---|---|
| row validation | `core.load_manifest` — a malformed manifest raises from there, unchanged |
| discovery and id safety | `core.find_audio`, via the `Take.has_audio` that `load_manifest` already resolved |
| confinement and atomic writing | `synthesis_session.ReaderSynthesisSession` |

Tests assert this module defines no `find_audio`, no `is_safe_utterance_id`,
no `AUDIO_SUFFIXES`, and calls neither `mkstemp` nor `os.replace`. The injected
`synthesize` is adapted to the session's round-trip shape rather than a second
writer being built.

### The writer is not injectable, on purpose

`generate_missing_takes(manifest_path, synthesize)` takes **no `writer`
argument**. An earlier version did, and that made every guarantee the write
carries — confinement, symlink refusal, byte validation, atomic replacement —
optional for any caller who passed their own. It also let the tests pass a
writer that returned a path without writing anything, so the suite was green
while proving less than it claimed.

`_default_writer` is private and is the only path bytes take to disk. Tests
that need to observe the write patch that boundary; one test asserts the
public signature has no `writer` parameter, and another proves the real
session is still in the path by planting a symlink where a take would be
written and checking the target survives.

Note when writing such a test: the symlink target must sit **outside** the
manifest directory. A link to a file inside it resolves cleanly, so
`find_audio` treats the take as already present and skips it — the write is
never attempted and the test proves nothing.

### Output location is not a parameter

Audio goes to `manifest_path.parent`. Naming and location together are what let
`find_audio` locate the result, so a caller cannot point them apart and end up
with audio the Reader cannot see. Two tests close the loop directly: after a
run, `find_audio` locates every take and `load_manifest` reports
`has_audio` for all of them.

### Resume means "the Reader can already find it"

An utterance is skipped when audio exists in **any** format `find_audio`
accepts — `.wav`, `.m4a`, `.mp3`, `.aiff`. So a real recording placed by hand
is honoured rather than overwritten by a generated one. Verified end to end: a
hand-placed `WARM-1.m4a` survived a run that generated the other two takes, and
the second run generated nothing at all.

### Failure leaves progress intact

A `synthesize` that raises stops the run and raises `ManifestRoundTripError`
naming the utterance, with the cause chained. Takes already written stay on
disk, and re-running resumes from the failure — which falls out of the resume
rule rather than needing bookkeeping. Unusable audio (empty or non-`bytes`) is
refused by the session before anything is written, and no `.partial` file
survives either path.

## Not implemented

Playback position, scrubbing, or re-recording. Playback is a one-shot `afplay`
subprocess per take.

### TDD evidence — manifest round trip

```bash
# red — no module yet
$ python3 -m pytest tests/voice_reader/test_manifest_roundtrip.py -q
ERROR — ModuleNotFoundError: No module named
        'voiceclonegpt.reader_app.manifest_roundtrip'

# green — the tests live in two modules, so both must be named
$ python3 -m pytest tests/voice_reader/test_manifest_roundtrip.py \
      tests/voice_reader/test_manifest_roundtrip_writer.py -q
50 passed
$ git diff --check
(clean)
```

Running only `test_manifest_roundtrip.py` reports **37** and silently skips
placement and the bypass protection; the figure for this seam is
**37 + 13 = 50**.

| Test module | Covers | Tests |
|---|---|---|
| `tests/voice_reader/test_manifest_roundtrip.py` | generation and ordering, resume across every accepted format, failure and re-run, input validation, and the borrowed-rule checks | 37 |
| `tests/voice_reader/test_manifest_roundtrip_writer.py` | placement beside the manifest, and that confinement, symlink refusal, byte validation, and atomic replacement cannot be bypassed | 13 |

| Run | Result |
|---|---|
| suite **without** this seam (baseline) | `1352 passed, 11 skipped` |
| suite **with** this seam | `1402 passed, 11 skipped` |

Worktree totals only — this checkout carries several other in-flight seams and
is behind `main`; the delta (+50) is the figure that travels. The baseline
must ignore **both** modules, or the writer tests are counted into it:

```bash
$ python3 -m pytest \
    --ignore=tests/voice_reader/test_manifest_roundtrip.py \
    --ignore=tests/voice_reader/test_manifest_roundtrip_writer.py -q
1352 passed, 11 skipped
```

**Version impact: MINOR — v0.2.0 capability.** Branch policy would place this
on `feature/v0.2.0-qwen-mlx-backend`; the current branch freeze keeps it on the
existing branch for orchestrator integration. No install, download, API call,
service call, or commit was made.
