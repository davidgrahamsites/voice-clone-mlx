---
type: module-contract
module: synthesis-runtime-registry
sequence: semantic
---

# synthesis — runtime adapter lookup

One job: answer "which adapter implements this runtime id". It performs no
loading policy, no conversion, no device selection, no downloads, and no
caching. Those belong to the modules named in
[`../../../docs/architecture/voice-model-lifecycle.md`](../../../docs/architecture/voice-model-lifecycle.md)
(`capability_selector`, `voice_model_loader`, `runtime_variant_converter`).

## Inputs

- A runtime id, as it appears in a bundle's `runtime_variants[].id`.
- Adapter factories registered by backend packages at import time.

Do NOT load: bundles, manifests, model weights, app state, or ingestion code.

## Process

1. A backend registers a factory under an explicit id.
2. A caller asks for that id and receives a **fresh** adapter instance.
3. An unregistered id raises `UnknownRuntimeError` naming the available ids.

Registration refuses to overwrite an existing id unless `replace=True` is
passed, so import order can never silently swap the backend behind a bundle.

An adapter satisfies the shape the lifecycle contract calls
`VoiceModelLoading` / `SpeechSynthesizing`:

```python
load(artifact_path, manifest) -> handle
synthesize(handle, text) -> bytes   # WAV
```

## Outputs

- Adapter instances, and `available()` — the sorted list of registered ids.

## The null runtime

`null_runtime.NullRuntime` is the only adapter **registered** today — an MLX
Qwen3-TTS adapter now exists (below) but is not in the registry. It returns a
valid, non-empty, deterministic mono PCM16 WAV of **silence**, with duration
derived from text length and clamped to `MAX_SECONDS`. It reads no model file;
`load` succeeds even when the artifact path does not exist.

It exists so the Reader path is runnable and testable before a voice model
exists. It imitates no one: every sample is zero and `IS_VOICE_MODEL` is
`False`. **Never flip that flag, and never present its output as the user's
voice** — a silent WAV must be visibly a placeholder, not a failed clone.

## Human check

1. `default_registry().available()` should list exactly `['null']` until a
   backend has been verified against a real model. If it lists more, confirm
   each entry is a genuine voice runtime someone has actually listened to.
2. Play a null-runtime WAV: it must be audible silence of a sensible length,
   not a zero-byte file and not a crash.
3. When the first real backend registers, re-read this contract: the registry
   must stay free of selection policy — choosing *which* runtime to use is the
   caller's decision, not this module's.

## The MLX Qwen3-TTS adapter

`mlx_qwen_runtime.MlxQwenRuntime` implements the same `load`/`synthesize`
shape for a **locally downloaded** Qwen3-TTS model via mlx-audio.

### It has never run against a real model

`IS_VERIFIED_AGAINST_REAL_MODEL = False`, and every one of its 71 tests injects
a fake loader and a fake model. **What is proven:** config validation, path
confinement, the call it makes to `model.generate`, sample→WAV conversion, and
the error contract. **What is not proven:** that Qwen3-TTS loads, that
`generate` has this signature in the installed version, or that the audio
sounds like anything. Do not describe this as working synthesis.

To actually verify it, on Apple Silicon:

```bash
pip install mlx-audio                     # not installed in this workspace
# download a Qwen3-TTS model to a local directory — the adapter never fetches it
```

then write a runtime config, load it, synthesize a sentence, and **listen**.
Only after that should `IS_VERIFIED_AGAINST_REAL_MODEL` be flipped, in the same
change that records who listened.

### Artifact: a runtime JSON config

```json
{
  "model_locator": "model",
  "ref_audio": "ref/neutral.wav",
  "ref_text": "This is the neutral reading.",
  "sample_rate": 24000
}
```

`ref_text` must be the **exact** transcript of `ref_audio` — Qwen's cloning
depends on it, so an empty value is refused rather than defaulted.

### No implicit downloads

mlx-audio will happily fetch a Hub id, so every locator is checked **before**
the loader is called:

| Rejected | Why |
|---|---|
| `hf://…`, `http(s)://…`, `s3://…`, `file://…` | anything containing `://` is remote |
| `mlx-community/Qwen3-TTS-0.6B` | a Hub repo id — it does not exist locally, so the existence check refuses it |
| `../outside`, `/etc` | resolved and confined to the config's own directory |
| a locator that is not a **directory** | a file where the model belongs is not a model; `exists()` alone would pass it to the loader |
| a locator that does not exist on disk | the model must already be downloaded |

`ref_audio` gets the same treatment and must be a real file. Tests assert the
injected loader records **zero** calls for each rejected case — the refusal is
before the load, not after.

### Audio conversion

Samples in `[-1.0, 1.0]` become mono 16-bit PCM via `wave` + `array` — no
numpy, no soundfile. Out-of-range values are **clamped, never wrapped**: a
wrapped sample is a loud click. Integers are treated as full-scale units, so
`1` means peak.

Model output is untrusted, so reading it is bounded and wrapped:

- **Never `list(audio)`.** Output may be a generator; materializing an
  unbounded one hangs the process instead of failing. `_to_sample_list` reads
  at most `MAX_OUTPUT_SAMPLES + 1` values — one past the cap is enough to know
  it is over.
- **Never a truthiness test** on the audio. mlx and numpy arrays raise
  "truth value of an array is ambiguous" from `__bool__`, so `audio or []`
  crashes on real output while passing against list fakes. The check is
  `audio is None`.
- `.tolist()` and iteration are provider code; both are wrapped, so a failure
  mid-stream becomes `SynthesisError` with the cause chained.
- **NaN and ±Infinity are refused**, not converted: `int(round(nan))` raises
  `ValueError` and `int(round(inf))` raises `OverflowError`, and neither should
  reach a caller as a raw exception.

## Local MLX Qwen bundle initializer

mlx_qwen_bundle.create_mlx_qwen_bundle(...) writes a manifest around a
Qwen3-TTS model that the caller already placed on disk. It never downloads,
copies, moves, converts, or loads a model. It writes only:

    <bundle>/bundle.json
    <bundle>/runtimes/mlx_qwen/config.json

The model directory and reference WAV must already be inside
<bundle>/runtimes/mlx_qwen/. The runtime confines paths to the config file's
directory, so accepting an asset elsewhere in the bundle would create a bundle
that validates but cannot load. Escaping paths and remote-looking locators are
refused before any write.

The manifest uses schema 1.0.0, runtime id mlx_qwen, backend qwen3-tts-mlx, an
allowed artifact kind, and a SHA-256 checksum over the exact config bytes.
Both destination files are checked before validation or writing; existing
files and symlinks are never overwritten. JSON is written through an
exclusively-created temporary sibling and os.replace, so a partial file cannot
be mistaken for a complete bundle. If the manifest write fails after the
config is written, the inert config is deliberately retained; the guard
requires a human to remove it before retrying.

Every path argument is coerced inside a typed-error guard. Invalid paths,
missing assets, traversal, outside symlinks, empty identity/text fields, bad
sample rates, and unsupported artifact kinds raise BundleInitError. The module
is detachable: it imports only the standard library and removing it does not
affect synthesis or the apps.

### TDD and review evidence

The initial red test was a missing-module error. The green split suites now
cover manifest shape, checksums, runtime-config contents, path confinement,
typed input refusal, no-overwrite behavior, atomic writes, and import
detachment:

    test_mlx_qwen_bundle_manifest.py: 16 passed, 1 documented skip
    test_mlx_qwen_bundle_safety.py: 70 passed
    full suite: 574 passed, 10 skipped

The documented skip is the shared-reader compatibility test when that module
is absent from a detached checkout; it becomes a real check when the shared
bundle reader is present.

### Not registered yet

`runtime_registry` still lists only `null`. Registering this adapter is a
separate change to `runtime_registry.py`, and it should not happen until the
real-model check above has been done — a registered runtime is one a caller can
select, and selecting an unverified backend would produce confident nonsense.

## Not implemented

Model conversion (PyTorch checkpoint → MLX), streaming, chunk caching, style
conditioning beyond the single reference clip, and any F5 comparison backend.

## Tests

- `tests/voice_reader/test_runtime_registry.py` — registration, duplicate
  refusal, typed unknown-id error, registry independence, WAV validity,
  determinism, duration bounds, silence, and an import-isolation check that the
  registry and null runtime pull in no backend, app, or ingestion code.
- `tests/voice_reader/test_mlx_qwen_runtime_config.py` (37) — config
  validation and path refusal: malformed JSON, wrong-typed fields, remote and
  traversal locators, `model_locator` that is not a directory. Each asserts the
  injected loader recorded **zero** calls.
- `tests/voice_reader/test_mlx_qwen_runtime.py` (18) — the generate call, the
  typed failures around it, and the lazy mlx-audio import.
- `tests/voice_reader/test_mlx_qwen_audio_conversion.py` (16) — int16
  conversion and clamping, bounded materialization of untrusted output,
  non-finite sample refusal, and wrapped `.tolist()`/iteration failures.
