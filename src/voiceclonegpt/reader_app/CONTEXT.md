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
  Generates takes through the shared round-trip contract. No UI toolkit.
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

## Synthesis session

`ReaderSynthesisSession(bundle_dir, runtime_id, runtime, output_dir)` turns
text into a WAV take on disk. `synthesize(text, output_name) -> Path`.

It owns **only** request validation, the call, and the write. Bundle
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
| `tests/voice_reader/test_synthesis_session.py` | 41 tests: the round-trip call, input validation, output confinement, typed failures, atomic replacement, and the lazy round-trip seam | no |

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

## Not implemented

Playback position, scrubbing, or re-recording. Playback is a one-shot `afplay`
subprocess per take.

