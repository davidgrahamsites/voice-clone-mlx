---
type: test-contract
module: tts-backend-seam-tests
sequence: semantic
---

# tests/voice_tts — what is actually proven about the TTS seam

Covers [`src/voiceclonemlx/tts/`](../../src/voiceclonemlx/tts/CONTEXT.md).
Read that contract first; this file says only what the tests establish and how.

## Inputs

`tmp_path` only. Every manifest, config, reference clip, and output file is
built inside the test's own temporary directory.

## The one thing to know

**Nothing here touches the network, mlx-audio, or a model.** No test
downloads, installs, loads weights, or reaches a service. `mlx_audio` is not
installed in this workspace and a test asserts that it is genuinely absent
(`importlib.util.find_spec("mlx_audio") is None`) — so the "backend
unavailable" path is exercised against reality rather than a mock of reality.

That is also the limit of the evidence. These tests prove the **seam** is
correct. They prove nothing about whether Qwen3-TTS loads, whether
`generate` has the assumed signature, or whether any of it sounds like a
human. See the blocker section of the module contract.

## Injection strategy

Every dependency is handed in as a constructor or factory argument. **Nothing
is monkeypatched at all** — the seam is injectable enough that no test needs
to reach into module globals, and a test that has to patch one is a signal the
seam has closed up.

| Fake | Stands in for | Shape |
|---|---|---|
| `FakeRuntime` | `MlxQwenRuntime` | `load(artifact_path, manifest)` / `synthesize(handle, text) -> bytes` |
| `BareStub` / `LoudStub` | `NullRuntime` | `synthesize(handle, text) -> bytes` |

`FakeRuntime` records its calls and can be constructed to raise on `load`
(`ImportError`, `RuntimeConfigError`), so unavailability is tested without
uninstalling anything.

## Files

### `test_backend_contract.py` — 21 tests

The protocol and the stub backend. Protocol conformance (`FakeBackend`
satisfies `TTSBackend`; an empty object does not); `SynthesisResult` populated
and frozen; a readable, mono, 16-bit, silent, deterministic WAV whose reported
duration matches the file; longer text gives a longer file; refusals (empty,
blank, and non-string text; a missing output directory) leaving nothing on
disk; and the no-god-object boundaries — imports only stdlib plus the one
module it reuses, and defines no `load`/`unload`/`enqueue`/`queue`/`warm_up`.

Two tests exist specifically because of a real break:

- `test_the_silence_settings_have_exactly_one_home` asserts `FakeBackend`
  defines **none** of `SAMPLE_RATE`, `MAX_SECONDS`, `MIN_SECONDS`,
  `CHARS_PER_SECOND`, `SAMPLE_WIDTH_BYTES`, `CHANNELS`. Mirroring another
  module's constant — even as a pointer — broke this seam once when that
  module reorganized its settings.
- `test_only_synthesize_is_required_of_a_stub` drives the backend with a stub
  carrying no attributes at all, proving the dependency is one method.

`test_duration_is_clamped` asserts **observed behaviour, not a constant**:
past the cap, more text stops producing more audio (200k and 400k characters
give equal durations). It never claims a particular cap exists, so it stays
true whether the stub clamps at 3 seconds or 300.

### `test_qwen_mlx_adapter.py` — 40 tests

The delegating adapter. Thinness (never imports `mlx_audio`; importing it does
not pull `mlx_audio` into `sys.modules`; defines neither `samples_to_wav` nor
`_to_sample_list`); the factory (config handed straight to the runtime,
missing mlx-audio and refused configs reported rather than raised, frozen
outcome); synthesis (handle and text forwarded, `out_path` returned and
written, duration measured from the WAV); the fixed-reference-clip rule
(configured clip accepted, `None` accepted, a different clip refused before
the runtime is called); and refusals leaving nothing on disk, including bytes
that are not a readable WAV.

`test_the_real_runtime_is_unavailable_in_this_workspace` builds a genuine
runtime config on disk and calls the factory with **no fakes at all**,
exercising the whole delegation chain down to the real mlx-audio import, and
asserts a clean unavailable outcome.

`TestOutcomeInvariant` (15) constructs `BackendOutcome` **directly**,
bypassing the factory. It covers both valid shapes; all four inconsistent ones
(available without a backend, available with a reason, unavailable without a
reason, unavailable with a backend); a blank `reason` (`""`, `"   "`, `"\n\t"`,
`" "`); a non-string `reason` (`42`, `0`, a list, a bare object); and that the
refusal quotes the offending value. Each raises `ValueError`. Testing the
factory alone would not establish the invariant — the factory is not the only
way to build one.

### `test_roundtrip.py` — 5 tests

That synthesized audio lands where the Reader already looks. Writes a
`script_session_1.jsonl` manifest, synthesizes `<utterance id>.wav` beside it,
and asserts `reader_app.core.find_audio` discovers each one and
`load_manifest` returns takes with audio — plus that an unsynthesized
utterance stays without. `reader_app` is imported **read-only** and its
discovery rule is never restated, so this test fails if the two ever drift.

It also holds the isolation check: an AST scan asserting no module in
`src/voiceclonemlx/` outside `tts/` imports `tts`, with a self-test proving
the scan can fail and does not false-positive on `mlx_audio.tts.utils`.

## Outputs

Nothing outside `tmp_path`. No run artifact, no fixture file, no cache.

## Human check

1. `PYTHONPATH=src python3 -m pytest tests/voice_tts/ -q` → **66 passed**
   (21 + 40 + 5). This count travels with the seam across checkouts — the
   integration mainline measures the same 66.
2. Full suite → **1508 passed, 11 skipped** in this worktree; with
   `--ignore=tests/voice_tts` → **1442 passed, 11 skipped**. Suite totals do
   **not** travel — the integration mainline reports different ones (see the
   labeled table in the module contract). Check the **delta**, which must
   equal the focused count, rather than matching a published total.
3. Grep this folder for `pip`, `install`, `http`, `requests`, or `download`.
   Today there is exactly one hit, and it is a docstring saying nothing here
   installs anything. Any hit in **executable** code violates the contract
   above and should be removed, not fixed.
4. Ask what a green run actually licenses you to say. The honest answer stays
   "the seam is wired correctly", never "voice cloning works".
