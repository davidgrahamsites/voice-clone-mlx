# studio_app — Voice Studio

**Job:** the launchable app shell and application-level composition seams for
Voice Studio. Load and record scripted material, or prepare reviewed free-speech
evidence for the existing dataset contract.

**Reads:** a `.txt` source file chosen by the user.
**Does:** parses it via `ingestion.parsers`, generates scripts via
`recording.script_generator`. It builds no ingestion or recording logic of its
own — if behaviour is missing, it belongs in those modules, not here.
**Writes:** `script_session_N.md` and `script_session_N.jsonl` into a chosen
output directory (written by `recording.script_generator`, not by this folder),
plus append-only WAV/JSON takes through `recording.capture` after Record.
**Human check:** `PYTHONPATH=src python3 -m voiceclonegpt.studio_app` opens a
window; opening a `.txt` lists its sentences; Record starts one bounded local
take. Confirm the macOS permission prompt cannot appear before Record.

## Files

- `core.py` — `StudioSession`, all logic, imports no UI toolkit.
- `audio_acceptance.py` — pure `validate_single_speaker_segments`
  decision seam over diarization output; no audio, model, network, or UI.
- `free_speech_dataset_pipeline.py` — two-phase coordinator from a captured
  session plus speaker-runtime evidence to a reviewed `DatasetManifest` JSON
  document; runtime, measurement, split, and dataset rules remain in their
  owning modules.
- `ui.py` — `StudioWindow` (Tk widgets only) plus `build_app()` / `main()`.
- `__main__.py` — `python3 -m voiceclonegpt.studio_app`.

## Seams

- `StudioWindow(master, session=..., choose_file=...)` — inject a session and a
  file-picker stub; constructing the window never starts an event loop.
- `build_default_session(recording_dir=..., capture_factory=...)` — the
  headless composition seam. It constructs the optional provider but does not
  call it. The default recording directory is the existing `~/Music`, falling
  back to the existing home directory; app launch creates nothing.
- Bus integration attaches at `shared.integration_seam.set_event_sink`. This
  folder emits `script_loaded`, `script_generated`, and `record_requested`, and
  never imports `voiceclonegpt.bus` directly.
- `prepare_free_speech_dataset(...) -> DatasetReviewBundle` delegates candidate
  admission to `alignment.free_speech_plan` and transcription to
  `alignment.whisper_runner`. Rejected spans remain reviewable and never reach
  the runner.
- `finalize_free_speech_dataset(...) -> DatasetComposition` requires one named,
  timed human decision per prepared transcript. It delegates alignment rows,
  session-exclusive splits, clip measurements, and dataset rows to their
  established domain seams; rejected candidates never reach the row builder.

## Boundaries

Must not import `reader_app`. May import `shared/`, `ingestion/`, `recording/`.
Deleting this folder must leave every other module working.

The macOS provider is also optional: `ui.py` imports it only inside the default
composition factory and falls back to the unavailable-recording session when
the provider module is absent. Permission and device access live entirely in
the provider and occur only when `StudioSession.record_line` calls it after an
explicit Record action.

The free-speech composer imports only public provider-neutral contracts. It
opens no audio, loads no model, calls no service, writes no artifact, and is not
imported by its producers or consumers. Deleting it leaves recording,
alignment, dataset, training, Voice Reader, and the Studio recording shell
importable.

## Tests and evidence

`tests/voice_studio/test_studio_app.py`.
`tests/voice_studio/test_free_speech_dataset_pipeline.py` uses injected local
fakes. Confirm an overlap or non-owner span is retained with its full original
range and reason while the fake transcription runner and dataset row builder
both remain uncalled.

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

Interactive stop, input-device selection, level meters, and recording-folder
selection. The first provider records one fixed-duration take from the macOS
default input device.
