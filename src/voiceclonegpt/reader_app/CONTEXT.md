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
- `ui.py` — `ReaderWindow` (Tk widgets only) plus `build_app()` / `main()`.
- `__main__.py` — `python3 -m voiceclonegpt.reader_app`.

## Seams

- `Player(command=...)` — the playback binary, `afplay` by default.
- `ReaderWindow(master, player=..., choose_file=...)` — inject a player and a
  file-picker stub; constructing the window never starts an event loop.
- Bus integration attaches at `shared.integration_seam.set_event_sink`. This
  folder emits `manifest_loaded` and `playback_started`, and never imports
  `voiceclonegpt.bus` directly.

## Boundaries

Must not import `studio_app`. May import `shared/`. Deleting this folder must
leave every other module working.

## Tests and evidence

Two modules, split by responsibility:

| File | Covers | Needs a display? |
|---|---|---|
| `tests/voice_reader/test_reader_manifest_security.py` | manifest loading, row schema, unsafe ids, traversal and symlink confinement | no — pure logic and temp files, runs in ~0.1 s |
| `tests/voice_reader/test_reader_app.py` | `Player`, `ReaderWindow`, and the headless-display gate | only the window tests, and only under the opt-in below |

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

