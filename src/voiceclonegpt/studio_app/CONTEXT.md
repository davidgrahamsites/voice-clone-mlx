# studio_app — Voice Studio

**Job:** the launchable app shell for recording. Load a source file, review the
lines that will be recorded, generate a recording script, and (eventually)
record takes.

**Reads:** a `.txt` source file chosen by the user.
**Does:** parses it via `ingestion.parsers`, generates scripts via
`recording.script_generator`. It builds no ingestion or recording logic of its
own — if behaviour is missing, it belongs in those modules, not here.
**Writes:** `script_session_N.md` and `script_session_N.jsonl` into a chosen
output directory (written by `recording.script_generator`, not by this folder).
**Human check:** `PYTHONPATH=src python3 -m voiceclonegpt.studio_app` opens a
window; opening a `.txt` lists its sentences; Record reports that capture is
not implemented yet.

## Files

- `core.py` — `StudioSession`, all logic, imports no UI toolkit.
- `audio_acceptance.py` — pure `validate_single_speaker_segments`
  decision seam over diarization output; no audio, model, network, or UI.
- `ui.py` — `StudioWindow` (Tk widgets only) plus `build_app()` / `main()`.
- `__main__.py` — `python3 -m voiceclonegpt.studio_app`.

## Seams

- `StudioWindow(master, session=..., choose_file=...)` — inject a session and a
  file-picker stub; constructing the window never starts an event loop.
- Bus integration attaches at `shared.integration_seam.set_event_sink`. This
  folder emits `script_loaded`, `script_generated`, and `record_requested`, and
  never imports `voiceclonegpt.bus` directly.

## Boundaries

Must not import `reader_app`. May import `shared/`, `ingestion/`, `recording/`.
Deleting this folder must leave every other module working.

## Tests and evidence

`tests/voice_studio/test_studio_app.py`.

### Headless display seam

Tk tests use `studio_tk_root`, defined in this app's own test module. It skips
**only** on a recognized display failure and re-raises any other `TclError`, so
a real widget bug cannot hide as an environment skip.

`tkinter.Tk` is constructed in exactly one place, `_open_hidden_root`, behind an
**explicit opt-in**:

```bash
python3 -m pytest tests/voice_studio/test_studio_app.py                       # UI tests skipped
VOICECLONEGPT_RUN_UI_TESTS=1 python3 -m pytest tests/voice_studio/test_studio_app.py   # UI tests run
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
   Reader would make Studio's tests fail when Reader is removed. Each app owns
   its display seam so neither can break the other.
2. **No silent fallback.** An earlier version named it `tk_root`, which shadowed
   the permissive fixture in `tests/conftest.py`. Under that name, deleting or
   misspelling the local fixture would have quietly fallen through to the
   conftest one, which skips on *any* `TclError` — turning a real widget failure
   into a green "skipped" run. With an app-specific name a missing fixture is a
   collection error instead.

Voice Reader has the mirror-image fixture, `reader_tk_root`. The two bodies are
similar by nature — both open a hidden Tk root — but they are separate facts
about separate apps, not one fact stored twice.

No test now uses the `tk_root` fixture in `tests/conftest.py`; retiring or
narrowing it needs a scope covering that file.

Current runs: **314 passed, 9 skipped** for the default full suite (the 9 are
the real-window tests across both apps), and **323 passed, 0 skipped** with
`VOICECLONEGPT_RUN_UI_TESTS=1`.

## Audio acceptance gate

validate_single_speaker_segments(segments, target_speaker=...) returns an
immutable AcceptanceResult for diarization segments. It decodes no audio,
runs no model, opens no file, and reaches no network. The diarizer that
produced the segments and the caller that handles rejected clips remain
separate modules.

The rule is intentionally strict: one overlap rejects the entire clip, and
any speaker label other than the target rejects the entire clip, even when
that speaker is far away in time. Gaps are allowed as silence. Touching
boundaries are adjacent, not overlapping. Malformed timing or speaker data
gets only the malformed reason; an empty input gets empty. Invalid target
speaker input raises ValueError because that is a caller error.

The frozen result contains sorted, normalized segments only when accepted.
Input is never mutated. The module is standard-library only; focused tests
also enforce that it imports no audio, model, network, subprocess, or UI
packages.

TDD evidence:

    test_audio_acceptance.py: 56 passed
    full suite after integration: 630 passed, 10 skipped

## Not implemented

Audio capture. `StudioSession.record_line` is a placeholder that returns a
message; it records nothing.
