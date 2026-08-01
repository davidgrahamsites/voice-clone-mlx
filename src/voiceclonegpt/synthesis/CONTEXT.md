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

## Local bundle initializer

`mlx_qwen_bundle.create_mlx_qwen_bundle(...)` describes a Qwen3-TTS model the
caller has **already placed on disk** as a loadable bundle. It writes exactly
two files and touches nothing else:

```text
<bundle>/bundle.json
<bundle>/runtimes/mlx_qwen/config.json
```

It never downloads, copies, moves, converts, or loads a model. If an asset is
not already where it belongs, that is an error — not something this module
fixes for you. Stdlib only; it imports nothing from the rest of the package, so
deleting it leaves the runtime adapter and both apps working.

### Assets must sit inside the runtime variant directory

Assets must be inside `<bundle>/runtimes/mlx_qwen/`, not merely inside the
bundle. That is stricter than "self-contained" for a concrete reason:
`mlx_qwen_runtime` confines config paths to **the config file's own
directory**, so a model at `<bundle>/model` would produce a manifest that
validates but cannot load. Rejecting it here fails at authoring time with a
message naming the constraint, instead of at synthesis time.

Symlinks are resolved before the check, so a link pointing outside is refused
and a link inside the variant directory is fine.

### Manifest shape

`bundle_schema_version` `1.0.0`; identity fields `bundle_id`, `voice_id`,
`model_version`; `source_model.artifact_kind` from the three supported kinds;
one `runtime_variants` entry with id exactly `mlx_qwen`, backend
`qwen3-tts-mlx`, a **relative** `artifact` path, and a SHA-256 over the exact
config bytes written.

The artifact-kind list is duplicated from `shared.model_bundle.ArtifactKind`
rather than imported — this module stays detachable, and the shared enum ships
on the model-roundtrip branch. **Keep the two in step**; a test asserts the
manifest satisfies every field `shared.bundle_reader.read_bundle` requires.

### Not overwritten, written atomically — with one honest limit

**Both outputs are checked before anything is validated or written**:
`bundle.json` *and* `runtimes/mlx_qwen/config.json`. Checking only the manifest
was not enough — a hand-edited runtime config can exist without a manifest, and
a rerun would have silently destroyed it. A symlink at either path counts as
existing, so it is refused rather than followed. Remove the file deliberately
to re-initialize.

Every argument that names a path is coerced inside a guard, so `None` or a
number raises `BundleInitError` naming the field rather than a raw `TypeError`
from `Path()`.

Both JSON files are written
through an exclusively-created sibling (`tempfile.mkstemp`, `O_EXCL`) and
`os.replace`, so no half-written JSON is observable and a planted temp path
cannot redirect the write.

**Limit:** the config is written before the manifest. If the manifest write
fails, the runtime config remains on disk. It is inert — without `bundle.json`
nothing will read it — and it is deliberately *not* deleted, because this
module refuses to remove files it cannot prove it created.

**A retry will not clear it.** The config guard above refuses any run where
`runtimes/mlx_qwen/config.json` already exists, and it cannot tell a leftover
from a hand-edited one. So after a failed manifest write you must **delete the
leftover config deliberately** before re-running. That is the intended
trade-off: refusing to guess costs one manual step, and never silently
destroys a config someone wrote by hand.

(An earlier version of this file said a retry would overwrite the leftover.
That was true before the config guard and is not true now.)

### TDD evidence

```bash
# red — no module yet
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle.py -q
ERROR — ModuleNotFoundError: No module named
        'voiceclonegpt.synthesis.mlx_qwen_bundle'

# green
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle.py -q
66 passed, 1 skipped in 0.24s
$ python3 -m pytest -q
554 passed, 10 skipped in 5.51s
$ git diff --check
(clean)
```

That single 405-line module was then split in two — ICM flags modules that
large — with the test ids diffed before and after to prove nothing moved but
the file boundary:

```bash
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_manifest.py -q
16 passed, 1 skipped
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_safety.py -q
50 passed
$ python3 -m pytest -q
554 passed, 10 skipped
```

Review then found two defects, both fixed test-first:

```bash
# red — an existing runtime config was silently overwritten, and a non-path
# argument leaked a raw TypeError
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_safety.py -q
14 failed, 50 passed in 0.81s

# green
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_safety.py -q
64 passed in 0.41s
$ python3 -m pytest -q
568 passed, 10 skipped in 5.65s
$ git diff --check
(clean)
```

A follow-up review found `bundle_dir` still coerced unguarded — the same
`Path(None)` leak, in the one argument the earlier fix missed:

```bash
# red
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_safety.py -q
6 failed, 64 passed in 0.68s

# green
$ python3 -m pytest tests/voice_reader/test_mlx_qwen_bundle_safety.py -q
70 passed in 0.46s
$ python3 -m pytest -q
574 passed, 10 skipped
```

A regression test writes `{"ref_text": "hand tuned by a human"}` into the
config, calls the initializer, and asserts those exact bytes survive and no
manifest appears.

The one skip is `test_read_bundle_accepts_the_manifest`:
`voiceclonegpt.shared.bundle_reader` is not in this checkout (it ships on
`fix/v0.6.2-model-roundtrip-contract`), so **the manifest has not actually been
read back by the shared reader**. A companion test asserts every field that
reader requires, and the skip converts to a real check the moment the branches
meet. Re-run it then before trusting the bundle end to end.

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
- `tests/voice_reader/test_mlx_qwen_bundle_manifest.py` (16 + 1 skipped) —
  manifest shape, checksum, runtime-config contents, and compatibility with
  the shared reader.
- `tests/voice_reader/test_mlx_qwen_bundle_safety.py` (70) — asset
  confinement, input validation (including non-path `bundle_dir`, `model_dir`,
  and `ref_audio`), no-overwrite for **both** outputs, atomic writes, and
  detachability.
- `tests/voice_reader/test_mlx_qwen_audio_conversion.py` (16) — int16
  conversion and clamping, bounded materialization of untrusted output,
  non-finite sample refusal, and wrapped `.tolist()`/iteration failures.

## Runtime readiness

`readiness.check_runtime_readiness(bundle_dir, *, runtime_id) ->
ReadinessReport` answers one question: **could a real model round trip run
here, and if not, what exactly is missing?**

It installs nothing, downloads nothing, imports no backend, loads no model, and
writes no file. It reports on the world; it does not change it.

### Every blocker at once, not the first one

`ReadinessReport(ready, blockers, checked)` lists **all** the problems, so the
answer is a checklist rather than a first-failure. Run in this workspace today:

```text
ready:    False
blockers: ('dependency_missing', 'model_absent', 'runtime_not_registered')
checked:  ('dependency', 'registration', 'config')
```

| Blocker | Meaning |
|---|---|
| `dependency_missing` | `mlx_audio` cannot be located |
| `runtime_not_registered` | no caller could select this runtime id |
| `bundle_unreadable` | the bundle reader refused it |
| `config_invalid` | the runtime config is missing, malformed, or otherwise refused |
| `model_absent` | the config names a model that is not present locally |
| `reference_absent` | the config names a reference clip that is not present |

### Nothing is validated twice

| Question | Answered by |
|---|---|
| is the backend installed? | `importlib.util.find_spec` — locates without importing |
| can this runtime be selected? | the registry — `default_registry().available()` here, `RuntimeRegistry.ids()` on `main` (see below) |
| is the bundle valid? | `shared.bundle_reader.read_bundle` |
| is the config usable? | `MlxQwenRuntime.load`, with a **probe loader that raises before any model opens** |

### Two registry APIs, one question

`main`'s `runtime_registry` and this worktree's have diverged: `main` exposes
`RuntimeRegistry.ids()` and **no** shared `default_registry()`, while this
checkout exposes `default_registry()` and `available()`. Integration surfaced
the mismatch as an `ImportError`.

`_registered_runtimes` therefore imports `default_registry` optionally, falls
back to `RuntimeRegistry`, and calls whichever lister the object has. The
question asked is identical and the list is never restated here — a test
asserts this module hard-codes no runtime id and defines no `register`.

Where there is no shared registry, a freshly built one is **empty**, so
`runtime_not_registered` is reported. That is the honest answer rather than a
guess: on `main` nothing is registered until a composition root does it, so no
runtime is selectable process-wide. Verified by executing the module against a
stand-in exposing only `main`'s surface.

**This compatibility shim should collapse to one import once the two registries
are reconciled.** It is a second home for nothing — only for *how to ask*.

The probe is the trick worth remembering: reaching the loader *is* the signal
that every config rule passed, so validity is established by running the real
validation rather than restating it, and `ProbeReached` guarantees no model is
ever opened.

**The probe is private and not overridable.** An earlier version exposed it as
`probe=`, which let a caller hand in a loader that genuinely opens a model —
turning a readiness *check* into a model load, the one thing this module
promises never to do. Tests assert the public signature has no `probe`
parameter, that passing one raises `TypeError`, and that `_probe` always
raises; tests needing to observe the call patch the private boundary.

`model_absent` and `reference_absent` are derived from which config field the
runtime named in its refusal. Those field names are part of the config
contract rather than incidental prose, but it is still a coupling — anything
unrecognized stays the general `config_invalid` rather than being guessed at.

### `ready` is not merely "no blockers"

`ready` requires that every check in `REQUIRED_CHECKS` actually **ran**. Where
`shared.bundle_reader` is absent — it ships on the model-roundtrip branch — the
bundle check is omitted from `checked` and the report is not ready, with no
misleading blocker. An empty `blockers` with `ready` False means *something
could not be verified*, which is not the same as being fine.

### What this does not do

It does not close the gap. Real synthesis still needs `mlx-audio` installed and
a Qwen3-TTS model downloaded on Apple Silicon, and
`IS_VERIFIED_AGAINST_REAL_MODEL` stays `False` until a human has listened. This
check makes the gap **legible** and turns it into a checklist — nothing more.

### TDD evidence

```bash
# red — no module yet
$ python3 -m pytest tests/voice_reader/test_readiness.py -q
ERROR — ModuleNotFoundError: No module named
        'voiceclonegpt.synthesis.readiness'

# green
# the tests live in two modules, so both must be named
$ python3 -m pytest tests/voice_reader/test_readiness.py \
      tests/voice_reader/test_readiness_probe.py -q
40 passed
$ git diff --check
(clean)
```

Running only `test_readiness.py` reports **36** and silently skips the probe
protection; the figure for this seam is **36 + 4 = 40** (was 35 + 4 = 39 before
the registry-compatibility test was added).

| Test module | Covers | Tests |
|---|---|---|
| `tests/voice_reader/test_readiness.py` | the all-clear path, every blocker in isolation and in combination, the skipped-check case, the report contract, purity, and that the registry is asked through whichever API the checkout ships | 36 |
| `tests/voice_reader/test_readiness_probe.py` | that the probe is private, cannot be overridden, and always raises — the guarantee that no model is ever loaded | 4 |

| Run | Result |
|---|---|
| suite **without** this seam (baseline) | `1402 passed, 11 skipped` |
| suite **with** this seam | `1442 passed, 11 skipped` |

Worktree totals only; the delta (+40) is the figure that travels. The baseline
must ignore **both** modules:

```bash
$ python3 -m pytest \
    --ignore=tests/voice_reader/test_readiness.py \
    --ignore=tests/voice_reader/test_readiness_probe.py -q
1402 passed, 11 skipped
```

`tests/voice_reader/test_readiness.py` (36) — the all-clear path, each blocker
in isolation, several at once, unknown runtime reported rather than raised,
sorted unique blockers from the fixed vocabulary, the skipped-check case, the
frozen report, purity including no writes, no network, and no import of the
backend, and registry compatibility. `tests/voice_reader/test_readiness_probe.py`
(4) — the probe protection.

**Version impact: MINOR — v0.3.0 capability.** Branch policy would use
`feature/v0.3.0-runtime-readiness`; the current freeze keeps this on the
existing branch. The seam itself performs no install, download, API call, or
service call; it was integrated on the frozen branch after review.
