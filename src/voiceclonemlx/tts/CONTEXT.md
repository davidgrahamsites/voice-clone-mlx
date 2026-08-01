---
type: module-contract
module: tts-backend-seam
sequence: semantic
---

# tts — one line of text in, one audio file out

One job: **given text and a place to put it, produce an audio file and say
what was produced.** Choosing a backend, loading models, chunking, batching,
retrying, queueing, publishing to the data bus, and deciding where session
audio lives are all somebody else's job. This module holds no state between
calls.

## Inputs

- `text` — one non-blank line to speak.
- `ref_audio_path` — the reference clip, or `None`. See "the clip is fixed at
  load time" below; it is not a free choice.
- `out_path` — where the audio goes. Its **directory must already exist**;
  this module never creates one, because inventing a folder hides a caller
  passing the wrong session path.

Do NOT load here: bundles, manifests, model weights, app state, ingestion
code, or bus contracts.

## Process

```python
synthesize(text, ref_audio_path, out_path) -> SynthesisResult
```

`SynthesisResult` is frozen: `out_path`, `duration_s`, `backend_name`.

`duration_s` is **measured from the audio itself**, never estimated, so a
reported duration cannot disagree with the file. That measurement doubles as
the validity check: backends measure the bytes *before* writing them, so a
failed generation or unusable output leaves **nothing on disk**.

Order of every `synthesize` call: refuse bad text → refuse a missing output
directory → refuse a mismatched reference clip → generate → measure → write.

## Outputs

- One WAV file at `out_path`, named by the caller. The round-trip test names
  it `<utterance id>.wav` beside the manifest, which is where
  `reader_app.core.find_audio` looks — that rule lives in the Reader and is
  **not restated here**.
- A `SynthesisResult`. Failures raise `SynthesisError`; unavailability does
  not raise (below).

## The two backends

| Module | Backend | Speaks? |
|---|---|---|
| `backend.py` | `FakeBackend` | No — silence |
| `qwen_mlx.py` | `QwenMlxBackend` | Not yet verified (see the blocker) |

### `FakeBackend` imitates no one

Every sample is zero and `IS_VOICE_MODEL` is `False`. **Never flip that flag,
and never present its output as the user's voice** — a silent WAV must stay
visibly a placeholder, not a failed clone.

It does not generate the silence itself. It delegates to
`synthesis.null_runtime.NullRuntime`, which already owns that job.

**The dependency is behavioural, not structural.** `FakeBackend` requires
exactly one method — `synthesize(handle, text) -> WAV bytes` — and reads *no
attribute* of the stub. Sample rate, speaking pace, and the duration clamp are
the stub's business: not copied here, and deliberately **not mirrored** here
either. An earlier version exposed `SAMPLE_RATE = NullRuntime.SAMPLE_RATE` as
"a pointer, not a copy". That was still a coupling to another module's
internal layout, and it broke the moment a branch reorganized those settings
into an instance attribute. A test now asserts this class defines none of
`SAMPLE_RATE`, `MAX_SECONDS`, `MIN_SECONDS`, `CHARS_PER_SECOND`,
`SAMPLE_WIDTH_BYTES`, or `CHANNELS`.

The stub is injectable, so any WAV-producing object can stand in.

### `qwen_mlx` delegates; it does not re-implement

`create_qwen_mlx_backend(config_path, manifest=None, runtime=None)` loads
`synthesis.mlx_qwen_runtime.MlxQwenRuntime` against a bundle's runtime config
and wraps it. The runtime is injectable.

Everything hard already has a home, so this module holds none of it:

| Concern | Owned by |
|---|---|
| lazy mlx-audio import | `mlx_qwen_runtime._default_loader` |
| refusing remote/hub locators | `mlx_qwen_runtime._confined_path` |
| `ref_text` / `sample_rate` validation | `mlx_qwen_runtime._validate_config` |
| bounded, clamped sample-to-WAV conversion | `mlx_qwen_runtime.samples_to_wav` |

Forking any of those would fork a rule that has already been hardened against
unbounded generators, non-finite samples, and out-of-range values that wrap
into audible clicks. A test asserts this module defines neither
`samples_to_wav` nor `_to_sample_list`.

**Importing this module never fails and never touches mlx-audio.** The import
happens inside the runtime's loader, at `load` time.

### Unavailability is an outcome, not an exception

`create_qwen_mlx_backend` returns a frozen
`BackendOutcome(available, backend, reason)`. A missing dependency, a missing
model, or a refused config comes back as `available=False` with a `reason`
naming what is wrong — never as a raised exception. Whether that is fatal is
the caller's decision, not this module's.

**Exactly one of `backend` and `reason` is set, enforced in `__post_init__`.**
`available=True` requires a backend and forbids a reason; `available=False`
requires a reason and forbids a backend. Anything else raises `ValueError`
when the outcome is *constructed*, so an inconsistent outcome cannot exist to
be read. This is checked on direct construction rather than trusted of the
factory: the factory is not the only way to build one, and a caller that
branches on `available` must find the matching field populated. Note the
consequence — a bare `BackendOutcome(available=False)` is refused, because
"unavailable" without a reason gives the caller nothing to act on or report.

**A reason must also be a nonblank string.** `""`, `"   "`, `"\n\t"`, and any
non-string are refused. `is not None` is too weak a test for a field whose
whole purpose is to be read by a person: a blank reason passes it and still
tells that person nothing, surfacing as an empty line in a log or a dialog
with no message. The refusal quotes the offending value so the caller that
produced it can be found.

### The reference clip is fixed at load time

The runtime resolves and confines its clip when it reads its config, so
`ref_audio_path` can only **confirm** that clip, never replace it. A
disagreement is **refused** rather than silently ignored, because ignoring it
would let a caller believe it had chosen a voice it had not. Pass `None` to
mean "use the configured clip".

## Isolation

`tts` is a leaf. **Nothing outside `tts/` imports it**, and
`tests/voice_tts/test_roundtrip.py` enforces that with an AST scan over every
module in `src/voiceclonemlx/` (plus a self-test proving the scan can fail and
does not false-positive on `mlx_audio.tts.utils`). The dependency arrow points
one way: `tts` → `synthesis`. Deleting this folder breaks nothing.

## The honest blocker — this is a wired seam, not working voice cloning

`qwen_mlx.IS_VERIFIED_AGAINST_REAL_MODEL` is `False`. **No real speech has
ever been produced through this module.** Every test injects a fake runtime or
a fake stub, `mlx_audio` is genuinely absent from this workspace (a test
asserts it), and `FakeBackend` emits silence. Do not describe this as working
synthesis, and do not flip the flag except in the same change that records who
listened.

Two things stand between here and real audio, **both awaiting user approval**;
neither was done, and no agent should do either unasked:

1. **Install `mlx-audio`.** The workspace's system Python 3.13 is externally
   managed, so this needs a virtualenv, not a bare `pip install`.
2. **Download Qwen3-TTS 0.6B weights** to a local directory. The adapter never
   fetches a model — it refuses hub ids and URL locators outright. Size is
   roughly 1–2.5 GB per the orchestrator's estimate; that figure has **not**
   been verified here.

`synthesis.readiness.check_runtime_readiness` already reports both as
blockers. Use it rather than guessing.

## Ownership

Authored by the Opus implementation worker under orchestrator `APPROVED WRITE`
scoping: only explicitly listed paths are writable, and everything else —
including `synthesis/` — is read-only. Fixes to this module require a fresh
scoped approval, and a fix needed *outside* the approved paths is reported as
a recommendation rather than applied.

## Version impact

**MINOR.** Additive: a new module behind an existing interface, changing no
seam, schema, bus message, or on-disk layout, and imported by nothing. The
slice is unreleased, so the API-compatibility fix described above was folded
into it rather than counted as a patch.

## TDD evidence

Tests were written first and failed for the right reason:

```bash
# red — no module yet
$ PYTHONPATH=src python3 -m pytest tests/voice_tts -q
ERROR — ModuleNotFoundError: No module named 'voiceclonemlx.tts'

# green
$ PYTHONPATH=src python3 -m pytest tests/voice_tts/ -q
66 passed
```

### Counts differ by checkout — read the label before quoting one

The seam's own test count travels; the suite total does not. Branches carry
different numbers of unrelated tests, so a single published total is a number
most checkouts will fail to reproduce. Both are recorded, labeled:

| Where | Focused (`tests/voice_tts/`) | Full suite |
|---|---|---|
| **This worktree**, measured here | `66 passed` | `1508 passed, 11 skipped` |
| same, `--ignore=tests/voice_tts` | — | `1442 passed, 11 skipped` |
| **Integration mainline**, measured there | `66 passed` | `1562 passed, 9 skipped` (`1571` including UI) |

**The two focused counts agree at 66**, which is the point: the seam's own
count is the same wherever it is checked out, so it is the number to quote.

The suite totals disagree — 1508 here, 1562 on the mainline — and that is
expected, not a defect. Each branch carries a different set of unrelated
tests. **Check the delta, never a total**: 1508 − 1442 = +66 locally, and the
equivalent subtraction on any checkout must also equal the focused count.

Totals are snapshots. Both rows went stale twice while this seam was being
built, each time because a change added tests after the figures were taken. If
a total here disagrees with what you measure, re-measure and trust your run —
the delta is the invariant, the totals are not.

`git diff --check` clean. Contract details for the tests themselves live in
[`../../../tests/voice_tts/CONTEXT.md`](../../../tests/voice_tts/CONTEXT.md).

## Human check

1. Play a `FakeBackend` output. It must be **audible silence** of a sensible
   length — not a zero-byte file, not a crash. If it ever sounds like a voice,
   something is very wrong.
2. Confirm `IS_VERIFIED_AGAINST_REAL_MODEL` is still `False`. If it is `True`,
   find the change that flipped it and confirm it names a human who listened.
3. Run `synthesis.readiness.check_runtime_readiness` before believing any
   claim that the real path works.
4. Grep this folder for `SAMPLE_RATE`, `MAX_SECONDS`, or `samples_to_wav`. The
   only hits should be prose. A real assignment means a rule grew a second
   home — the exact defect that broke this module once already.
