# VoiceCloneMLX real-voice round-trip procedure

## Purpose and current status

This manual describes how to move from a clean recording of the owner's voice
to a learned voice-model bundle that Voice Reader can load locally on an Apple
Silicon Mac.

The repository currently provides tested, modular contracts for recording,
transcription planning, speaker/overlap evidence, dataset review, bounded
training, source-model approval, MLX conversion, parity approval, immutable
bundle publication, Reader readiness, and WAV/MP3 output. It does **not** yet
provide one command that executes the whole lifecycle. Real dependencies,
model weights, a graphics-processor training provider, and real listening
approval are not installed or configured.

Do not call any of these a trained personal voice:

- a short reference-audio clone;
- the silent `NullRuntime` used by tests;
- a fake-backed passing test;
- an empty or metadata-only bundle; or
- a converted model that has not passed parity and listening review.

Completion means all of the following are true: a reviewed owner-only dataset
produced a learned checkpoint; that checkpoint was approved; its MLX conversion
passed the frozen parity set; a complete checksummed bundle was published;
Voice Reader loaded the exact declared runtime; and a human heard a non-silent
WAV or MP3 output that was intelligible and recognizably the owner. Reader can then
encode that verified WAV locally as MP3 when the output filename ends in
`.mp3`.

## Non-negotiable safety rules

1. Use only the owner's voice and material the owner is authorized to use.
2. Keep raw recordings local unless the owner explicitly approves a named
   provider and upload scope.
3. Reject an entire candidate clip if another voice overlaps any part of it.
   Do not use source separation to rescue it for training.
4. Never automatically approve a transcript, dataset row, checkpoint, runtime
   conversion, parity result, or final voice.
5. Never burst an application programming interface (API), model host, website,
   graphics-processor provider, or Git. Use one request or job at a time.
6. Stop on authentication, quota, rate-limit, checksum, path, or repeated
   execution errors. Do not silently retry or fall back to a different model.
7. Never overwrite raw recordings, accepted datasets, checkpoints, model
   versions, or published bundles.
8. Keep API keys, passwords, account numbers, home addresses, private URLs, and
   unreleased confidential text out of recordings and manifests.

## Required approvals before real execution

Record each approval in the run notes before performing the operation.

- [ ] Permission to create `/Users/appleadmin/Apps/VoiceCloneMLX/.venv`.
- [ ] Permission to install exact pinned MLX/Whisper dependencies.
- [ ] Acceptance of the exact Qwen, converter, and runtime licenses/revisions.
- [ ] Permission to download exact model assets to named local directories.
- [ ] Consent to record, transcribe, retain, and train on the owner's voice.
- [ ] A named remote GPU provider, maximum dollar cost, maximum runtime,
      retention period, deletion procedure, and approved upload contents.
- [ ] Permission to create run, checkpoint, conversion, bundle, and output files.
- [ ] Permission to play generated audio for listening review.
- [ ] Named human reviewers for dataset, source checkpoint, MLX parity, bundle,
      and final listening acceptance.

The required approval wording for an agent-run mutation is:

```text
APPROVED WRITE: <exact paths or external operation>.
Budget: <maximum amount or “no external spend”>.
Data allowed: <exact files or “no upload”>.
Rollback/deletion: <exact procedure>.
```

## Stage 0 — Freeze the experiment

Create a run record before recording or downloading anything.

Choose and write down:

- `voice_id`: a stable safe identifier, such as `david-narrator`;
- `run_id`: UTC date and purpose, such as `2026-08-01-pilot-01`;
- recording device, room, microphone distance, gain, and recorder settings;
- source model repository and exact revision;
- fine-tuning recipe and exact revision;
- converter and exact revision;
- MLX-Audio version/revision;
- frozen evaluation prompts and thresholds;
- GPU provider, budget, time limit, and deletion policy; and
- the people who will approve each human gate.

The intended artifact layout is:

```text
data/voices/<voice-id>/
  enrollment/
  runs/<run-id>/
    01_recording/
    02_transcription/
    03_speaker-evidence/
    04_alignment-review/
    05_dataset/
    06_training/
    07_source-evaluation/
    08_conversion/
    09_runtime-parity/
    10_bundle-publication/
  models/versions/<model-version>/
  promoted.json
```

This blank run tree is not generated automatically yet. Create it only after a
scoped write approval. Never reuse a prior run directory.

## Stage 1 — Prepare the native Apple Silicon environment

The Mac is Apple Silicon (`arm64`). `/usr/local/bin/python3` is an Intel
interpreter on this machine. The intended native interpreter is:

```text
/opt/homebrew/bin/python3.13
```

After explicit install approval, create a project-local environment:

```bash
cd /Users/appleadmin/Apps/VoiceCloneMLX
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -c 'import platform; print(platform.machine())'
.venv/bin/python -m pip install --require-hashes -r requirements-mlx.lock
```

The architecture check must print `arm64`. Do not continue if it prints
`x86_64`.

**Stop:** `requirements-mlx.lock` does not exist yet. Do not replace the locked
command with an unpinned `pip install`. A separate approved dependency task must
resolve compatible versions, record every transitive package/version/hash,
review the resulting lock, and add the file before this command can run. The
official MLX-Audio project currently documents its package and requirements at:
<https://github.com/Blaizzy/mlx-audio>. Do not start its web server; VoiceCloneMLX
uses local library/runtime seams.

Verify imports without loading a model:

```bash
.venv/bin/python -c 'import mlx; import mlx_audio; import mlx_whisper; print("imports ok")'
```

The native-launcher patch is prepared separately and intentionally unmerged.
Merge it only after `.venv/bin/python` exists and passes the architecture and
import checks. Otherwise both `.app` launchers would correctly refuse to run.

## Stage 2 — Build and smoke-test the local apps

Current source entry points are:

```bash
cd /Users/appleadmin/Apps/VoiceCloneMLX
PYTHONPATH=src .venv/bin/python -m voiceclonemlx.studio_app
PYTHONPATH=src .venv/bin/python -m voiceclonemlx.reader_app
```

Build the wrappers after the native-launcher change is integrated. `--replace`
deletes and recreates the two generated bundles, so first copy the existing
`apps/Voice Studio.app` and `apps/Voice Reader.app` to a dated backup directory
and obtain an explicit replace approval. Rollback is deleting the failed new
wrappers and restoring both backed-up bundles together.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_apps.py --replace
```

Expected bundles:

```text
apps/Voice Studio.app
apps/Voice Reader.app
```

Manual checks:

- [ ] Voice Studio opens from Finder.
- [ ] Voice Reader opens from Finder.
- [ ] Voice Studio requests microphone permission only after Record is clicked.
- [ ] Voice Reader does not request microphone permission.
- [ ] Both apps show a clear error when a required local dependency is missing.
- [ ] No model download starts during app launch.

## Stage 3 — Record the 30-minute pilot

Use [`data/scripts/voice_training_script_30_minutes.md`](../data/scripts/voice_training_script_30_minutes.md).
It is designed as two approximately fifteen-minute sessions so vocal fatigue
does not become part of the learned voice.

### Recorder and room

Audio terms used below:

- **Room tone:** the natural sound of the room while nobody speaks.
- **Gain:** the recorder's input-amplification setting.
- **dBFS:** decibels relative to the loudest digital level; zero dBFS is the
  clipping ceiling.
- **Clipping:** flat-topped distortion caused when the input is too loud.
- **WAV/PCM:** a WAV container holding uncompressed pulse-code-modulated audio.

1. Use lossless WAV/PCM at the recorder's highest reliable quality.
2. Keep one room, device, orientation, distance, gain, posture, and signal path.
3. Disable automatic gain, compression, reverb, and aggressive enhancement.
4. Aim for peaks near -12 to -6 dBFS without clipping.
5. Record thirty seconds of room tone at the beginning and end of each session.
6. Keep the untouched master. Processing always uses a derived copy.

### Reading protocol

1. Do not speak prompt IDs, headings, pause instructions, or notes.
2. At each style block, say exactly: `This is the <style> reading.`
3. Wait three seconds after the marker. Count silently: “one-Mississippi,
   two-Mississippi, three-Mississippi.”
4. Read each numbered prompt naturally and wait two seconds after the prompt's
   final word. Count silently: “one-Mississippi, two-Mississippi.” This pause
   does not repeat after every word, clause, or internal breath.
5. A long prompt or calibration passage may contain natural breaths at commas,
   semicolons, and sentence endings. Keep the recorder running; do not insert a
   deliberate two- or three-second gap at every internal punctuation mark.
6. If a word is wrong, stop, wait two seconds, and reread the whole prompt once.
6. If another person speaks over any part, continue only after a clean pause;
   the mixed candidate will be rejected in full.
7. Stop at a prompt boundary if the voice becomes dry, strained, breathy,
   rushed, unusually high/low, or materially quieter/louder. Rest five to ten
   minutes and restart with the calibration passage.

### Recording-ready gate

Before pressing Record:

- [ ] The approved `voice_id` and `run_id` are written in the run notes.
- [ ] The new run directory exists and contains empty numbered stage folders.
- [ ] No prior run directory or recording will be overwritten.
- [ ] Recorder, room, distance, gain, format, and operator are recorded.
- [ ] The script version/checksum is recorded.
- [ ] Consent and retention/deletion decisions are recorded.
- [ ] At least ten minutes of free recorder storage remains beyond the estimate.
- [ ] A brief level check shows clear speech with no clipping.

After scoped approval, create the empty run tree with separate explicit
`mkdir -p` commands for the paths listed in Stage 0. Confirm every directory is
under the intended `voice_id/run_id` before recording; never use a wildcard or
an already populated run.

### Immediate acceptance check

Before recording more material, inspect the pilot:

- [ ] no clipping, dropouts, heavy hiss, clicks, handling noise, or music;
- [ ] stable distance and tone from opening to closing calibration;
- [ ] marker phrases are clear and followed by silence;
- [ ] all speech is the owner;
- [ ] the raw WAV checksum is recorded; and
- [ ] the owner can listen comfortably without obvious room echo.

If the chain fails this pilot, fix the room/device procedure and rerecord. Do
not compensate by training longer on bad audio.

## Stage 4 — Copy, checksum, and inspect recordings

1. Copy the file from the recorder into `01_recording/raw/`.
2. Never process directly from removable media.
3. Compute and store SHA-256:

   ```bash
   shasum -a 256 /absolute/path/to/session.wav
   ```

4. Inspect the actual container and stream rather than trusting the recorder
   label:

   ```bash
   afinfo /absolute/path/to/session.wav
   ```

5. Record duration, channels, sample rate, bit depth/format, file size, checksum,
   recorder settings, room, device orientation, and clipping observations.
6. Make a derived mono working copy only if the selected backend requires it.
   Keep the original master unchanged.

Reject the complete source file and investigate if it is corrupt, truncated,
unexpectedly compressed, or recorded with an unknown automatic effect.

## Stage 5 — Enroll the owner and reject other voices

Enrollment identifies which anonymous diarized speaker is the owner; it is not
the training model.

1. Select 30–60 seconds of unquestionably clean owner-only audio for the first
   enrollment, preferably 2–5 minutes across normal neutral, warm, and energetic
   delivery for an all-day recorder.
2. Keep at least one clean owner clip held out from threshold calibration.
3. Record enrollment checksums, recorder identity, room type, and style.
4. Run voice activity detection to locate speech.
5. Run local diarization to mark anonymous speaker turns.
6. Run local speaker verification against the enrolled owner.
7. Produce immutable evidence for every candidate: source checksum, start/end,
   speaker label, owner score, overlap flag, backend/revision, threshold, and
   decision.

Decision policy:

- `accept`: verified owner and no overlap/non-owner interval;
- `review`: uncertain owner score or quality problem, never automatic training;
- `reject`: non-owner, any overlap, clipping, corrupted audio, or unsafe path.

If two voices overlap for even part of a candidate, reject that candidate in
full. Do not trim around the overlap and do not use source separation to make a
training clip.

## Stage 6 — Transcribe and align locally

The local Whisper resources are documented in
[`data/scripts/MLX_WHISPER.md`](../data/scripts/MLX_WHISPER.md). Prefer an
explicit local model path and record the runtime/model revision. Never allow a
transcription command to interpret a remote model locator.

For scripted sessions:

1. Transcribe the master/accepted spans with local MLX Whisper.
2. Match timestamps to known prompt IDs and spoken style markers.
3. Remove marker speech from training candidates.
4. Split candidates at prompt boundaries while retaining source offsets.
5. Compare recognized text with the known script.
6. Correct the transcript to exactly what was actually spoken; do not silently
   change the audio to match intended text.

For unscripted/all-day sessions:

1. Apply owner/overlap rejection before transcription.
2. Transcribe accepted owner-only candidates.
3. Segment into natural 2–12 second utterances.
4. Preserve the full rejected decision record.
5. Manually correct every transcript admitted to training.

Human alignment gate:

- [ ] Listen to every uncertain boundary and mismatch.
- [ ] Confirm no marker audio remains.
- [ ] Confirm no second voice, television, radio, or playback is present.
- [ ] Confirm transcript and punctuation describe what was spoken.
- [ ] Record reviewer, UTC time, source checksum, and decision.

## Stage 7 — Build the versioned dataset

Only accepted, reviewed, owner-only audio can become a dataset row.

1. Store sentence-level WAV clips under style-specific directories.
2. Store prompt text, style, source session, source offsets, checksums, speaker
   evidence, transcript/runtime revisions, reviewer, and review time.
3. Split by recording session—not random clips—so room/session drift is visible.
4. Reserve approximately 90% train, 5% validation, and 5% untouched final test.
5. Never tune thresholds or choose checkpoints using the final test set.
6. Generate a dataset manifest and checksum it.
7. Listen to a random sample of ordinary accepted rows and every low-margin row.

Dataset release gate:

- [ ] zero mixed-speaker or overlap candidates;
- [ ] zero missing transcripts or checksums;
- [ ] no train/validation/test session leakage;
- [ ] no clipping or obvious processing artifacts;
- [ ] style distribution and prompt coverage recorded;
- [ ] licenses/consent recorded; and
- [ ] named human acceptance recorded.

## Stage 8 — Establish a zero-shot baseline

Before spending money on training, synthesize a fixed evaluation set using the
unmodified Base model plus a clean owner reference clip. This is a baseline,
not the trained deliverable.

Freeze 10–20 prompts covering neutral narration, questions, numbers, technical
terms, warm delivery, long sentences, and names. Record prompt text/checksums,
reference audio/checksum, model/revision, settings, and output checksums. Listen
and record identity, intelligibility, pacing, pronunciation, noise, and style.

The fine-tuned model must beat this baseline on the intended narration use case
without unacceptable regressions.

## Stage 9 — Choose and pin the training recipe

Primary candidate: official Qwen3-TTS Base fine-tuning. Official source:
<https://github.com/QwenLM/Qwen3-TTS>.

Do not assume that a model advertised as fine-tunable works with the current
recipe. As of this procedure, public upstream issues report 0.6B dimensional
mismatch failures in the fine-tuning script. Therefore:

1. Pin an exact repository commit and model revision.
2. Read the matching official fine-tuning files.
3. Run a tiny synthetic/no-private-data smoke test on the chosen CUDA image.
4. If 0.6B fails at the pinned revision, do not improvise an architecture patch
   for the paid run. Choose a verified upstream fix or evaluate 1.7B with an
   appropriate GPU and new cost preflight.
5. Record recipe ID, repository commit, model revision, tokenizer revision,
   container image digest, CUDA/PyTorch versions, batch size, accumulation,
   precision, learning rate, epochs, checkpoint cadence, and seed.

F5-TTS remains a separately labeled personal/noncommercial comparison. Never
silently substitute it for Qwen, and never mix their artifacts or licenses.

## Stage 10 — Approve the remote GPU job

Complete this worksheet:

```text
Provider:
Region:
GPU type and VRAM:
Maximum hourly price:
Maximum total cost:
Maximum wall time:
Concurrency: 1
Dataset files approved for upload:
Model files approved for download/cache:
Checkpoint interval:
Remote retention period:
Deletion command/procedure:
Credential source (never stored in repo):
Stop conditions:
```

Before submission:

- [ ] current price and quota inspected;
- [ ] total worst-case cost calculated;
- [ ] one-job concurrency enforced;
- [ ] timeout and output-size cap enforced;
- [ ] no automatic retry or fallback;
- [ ] resumable checkpoint path defined;
- [ ] only approved dataset files included;
- [ ] secrets excluded from manifests and logs;
- [ ] shutdown and deletion procedure tested on an empty job; and
- [ ] exact immutable command plan reviewed.

Start one job. Stop immediately on authentication, quota, rate-limit, repeated
error, unexpected model download, cost-cap, timeout, or checksum failure.

## Stage 11 — Train and retrieve learned checkpoints

1. Upload only the approved dataset and required configuration.
2. Verify remote checksums match local checksums before training.
3. Execute the approved argument-vector command; never a shell-constructed
   command assembled from untrusted text.
4. Save checkpoints at the approved cadence.
5. Preserve the last valid checkpoint on interruption.
6. Do not retry automatically.
7. Retrieve the training manifest, logs, configuration, and candidate checkpoint.
8. Verify every downloaded checksum.
9. Shut down the GPU immediately after required artifacts are retrieved.
10. Verify billing stopped and apply the approved remote deletion policy.

A valid training result must be a nonempty learned artifact such as
`fine_tuned_full` or `fine_tuned_adapter`; `reference_clone` is not accepted.

## Stage 12 — Evaluate and promote the source checkpoint

1. Run the frozen acceptance prompts with the source/PyTorch checkpoint.
2. Produce one checksum-bound preview for every prompt.
3. Record required objective metrics and thresholds.
4. A human listens to every preview.
5. Reject the checkpoint if any mandatory threshold fails, any preview is
   missing, or provenance/checksums do not bind to the exact training run.
6. Record named, time-zone-aware approval only after listening.
7. Build immutable source-release metadata. This returns copy instructions; it
   does not automatically copy, convert, publish, or promote anything.

## Stage 13 — Convert the accepted source model to MLX

1. Pin the converter identity/revision and MLX-Audio version.
2. Bind the accepted source-release and checkpoint checksums.
3. Bind the exact tensor/data-type mapping, target format, quantization, base
   model, tokenizer, and frozen parity-set checksum.
4. Execute one approved conversion on the chosen host.
5. Compute checksums for every converted payload.
6. Produce a conversion manifest whose status remains `pending`.

Conversion success does not mean runtime acceptance. Never register or publish
the candidate at this stage.

## Stage 14 — Run source-versus-MLX parity review

For every frozen prompt:

1. Generate source and MLX candidate WAV files with bound settings.
2. Verify prompt, source WAV, candidate WAV, source release, candidate,
   conversion, and parity-set checksums.
3. Record intelligibility error, alignment error, duration delta, and real-time
   factor using frozen thresholds.
4. Keep the result `pending_listening` even if all metrics pass.
5. A named reviewer listens to every source/candidate pair.
6. Reject incomplete listening, failed metrics, identity drift, instability,
   artifacts, wrong pacing, or checksum mismatch.
7. Record a time-zone-aware accepted decision bound to the exact report.

## Stage 15 — Publish the immutable bundle

Publication requires an accepted parity report.

1. Verify complete source, training, dataset, converter, runtime, evaluation,
   license, notice, and reference provenance.
2. Require one unambiguous default consented reference.
3. Refuse absolute paths, traversal, symlinks, missing/unlisted payloads, case
   collisions, and checksum mismatches.
4. Copy into a new sibling staging directory.
5. Recheck every copied payload.
6. Write `bundle.json`.
7. Write sorted `checksums.sha256` covering every bundle file except itself.
8. Verify the completed staging tree.
9. Atomically rename staging to a destination that does not already exist.
10. On failure, remove staging and leave no destination.

Reader must reject older/incomplete bundles without `checksums.sha256`; rebuild
them deliberately rather than bypassing verification.

Inspect before promotion:

- [ ] bundle ID/version and learned artifact kind;
- [ ] exact runtime declaration;
- [ ] all payload checksums;
- [ ] source/conversion/parity approval checksums;
- [ ] licenses, notices, references, and consent;
- [ ] no symlinks or unexpected files; and
- [ ] no existing version was overwritten.

Updating `promoted.json` is a separate atomic, human-approved action. Bundle
publication never changes it automatically.

## Stage 16 — Verify and enable Voice Reader

Reader production currently registers only `null`, which generates silence.
The MLX runtime has a deliberate verification flag set to false. Do not change
either fact until the real local listening gate passes.

1. Place the accepted immutable bundle in the approved local versions directory.
2. Start Voice Reader with Voice Studio unavailable to prove consumer
   independence.
3. Select the exact bundle and its manifest-declared runtime.
4. Run readiness. It must verify dependency, registration, bundle, config,
   model payload, reference, and checksums before loading.
5. Enable the real runtime only in a versioned change after human approval.
6. Enter a fixed test paragraph and generate to a new output directory.
7. Verify WAV container, sample rate, frames, duration, checksum, and that the
   samples are not all zero.
8. Play the file after playback approval.
9. Listen for owner identity, intelligibility, cadence, pronunciation,
   expressiveness, noise, clicks, truncation, repetition, and long-form drift.
10. Record bundle ID, runtime ID/version, model/config/reference checksums,
    prompt checksum, device, generation settings, output checksum, listener,
    time, and decision.

Only after this gate may the MLX runtime be registered and placed in Reader's
human-verified allowlist.

## Stage 17 — Final verification and release evidence

Run the repository gates:

```bash
cd /Users/appleadmin/Apps/VoiceCloneMLX
PYTHONPATH=src .venv/bin/python -m pytest -q
python3 /Users/appleadmin/.codex/skills/icm-check/scripts/icm_check.py .
git diff --check
PYTHONPATH=src .venv/bin/python scripts/build_apps.py --replace
```

Final acceptance checklist:

- [ ] native arm64 environment and pinned dependencies recorded;
- [ ] raw recording checksum and consent recorded;
- [ ] owner enrollment calibrated with held-out evidence;
- [ ] all mixed/overlap clips rejected in full;
- [ ] exact reviewed transcripts and session-exclusive dataset splits;
- [ ] learned checkpoint and training lineage verified;
- [ ] source previews heard and source release approved;
- [ ] MLX conversion checksummed and parity-pending until review;
- [ ] every parity sample heard and accepted;
- [ ] immutable complete bundle published without overwrite;
- [ ] Reader loaded the exact bundle/runtime independently;
- [ ] output WAV is structurally valid, nonempty, and non-silent;
- [ ] the owner heard and accepted identity and narration quality;
- [ ] GPU stopped, billing verified, and remote data deleted per policy;
- [ ] both `.app` bundles rebuilt and launched; and
- [ ] tests, ICM, and diff checks pass.

## Recovery and rollback

- Recording failure: preserve the rejected source and reason; create a new
  session rather than overwriting.
- Transcript/alignment failure: correct review data or rerun only the affected
  local stage; do not change the raw master.
- Training interruption: resume only from the latest verified checkpoint;
  never invent history or overwrite earlier checkpoints.
- Cost or provider failure: stop the job and billing; retain verified local
  evidence; do not auto-resubmit.
- Conversion/parity failure: keep the accepted source release; reject the MLX
  candidate; do not register it.
- Bundle publication failure: delete only the staging directory; the absent
  destination and previous versions remain unchanged.
- Reader failure: unregister only the rejected candidate version and restore
  the prior promoted pointer; never modify the immutable bundle in place.
- Privacy withdrawal: follow the recorded deletion map for raw files, derived
  clips, remote copies, checkpoints, conversions, bundles, and generated audio.

## Troubleshooting decisions

| Symptom | Stop and check | Do not do |
|---|---|---|
| App opens with Intel Python | Launcher path and `platform.machine()` | Install MLX into the Intel interpreter |
| MLX dependency missing | Project `.venv`, pinned install record | Reuse another app's virtual environment |
| Model attempts a Hub download | Runtime/config contains a remote locator | Allow an implicit download |
| 0.6B fine-tune shape mismatch | Pinned upstream recipe/revision | Patch dimensions during the paid run |
| Other voice or overlap | Speaker evidence and source interval | Trim or separate and accept the remainder |
| Transcript differs from speech | Human transcript correction | Change text to what was intended |
| Generated WAV is silent | Runtime registration, samples, config, bundle | Call the null runtime a success |
| Voice sounds like owner but pacing is bad | Baseline/parity prompts and training epochs | Approve identity alone |
| GPU job loops or cost rises | Stop condition, timeout, provider shutdown | Retry automatically |
| Bundle checksum fails | Source payload and staging copy | Edit checksum files by hand |
