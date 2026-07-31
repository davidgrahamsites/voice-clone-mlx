# Extracting the owner's voice from all-day recorder files

**Status:** research recommendation for the local Voice Studio ingestion stage  
**Intended software-version impact:** `PATCH` (`0.1.x`): this is a research/documentation artifact only; it does not change a public module seam, schema, model bundle layout, or runtime contract.  
**Integration branch:** the orchestrator should integrate this file on `fix/v0.1.1-voice-isolation-research` (the research worker did not switch branches or modify app code).  
**Privacy:** keep raw recordings and all derived audio local. Do not upload an all-day recording to a hosted transcription/separation service without an explicit, per-run choice.

## Short answer

The recorder cannot identify *your* voice. Its “noise reduction” changes the
signal and its voice-activated recording (VOR) starts or stops when **any**
sound crosses a threshold; neither feature is speaker recognition. Use the
recorder as a source of raw files, then run a local, staged pipeline:

1. keep the original file untouched and inspect its real codec/sample rate;
2. detect speech (VAD);
3. diarize speech into speaker turns;
4. identify the owner's turns with an enrolled voice embedding;
5. reject the entire candidate clip if another voice overlaps or appears in
   that clip; and
6. preserve the rejected source span only for audit, never for training.

This is **speaker filtering**, not a single magic “extract me” operation. It is
reliable when speakers take turns. The dataset rule is intentionally stricter:
if two voices occur anywhere in one candidate clip, the whole clip is clipped
from the dataset and rejected. We do not attempt to rescue overlap with source
separation.

## What the purchased Tonfarb A20 provides

The product manual identifies the device as the **Tonfarb A20**. The manual
describes dual built-in microphones, PCM recording up to 1536 kbps, MP3/WAV
output, five noise-reduction levels, adjustable gain, VOR, a 136 GB storage
configuration (8 GB internal + 128 GB TF card), and up to 68 hours of continuous
recording. See the [A20 manual](https://manuals.plus/asin/B0DCFDRD2K.pdf),
especially the feature and specification pages.

Important caveats:

* The manual is a retailer-hosted copy; treat the first transfer from the
  recorder as authoritative. Run `ffprobe` or macOS `afinfo` on a sample file
  and record channels, sample rate, sample format, codec, and clipping rate in
  the manifest. A label such as “1536 kbps PCM” is not enough to infer whether
  the file is mono/stereo or 16/24-bit.
* Prefer the highest-quality **WAV/PCM** setting for model data. MP3 is useful
  for listening and review but adds a lossy codec before diarization and voice
  cloning.
* If the menu permits it, use the lowest/noise-reduction setting for the source
  capture and apply enhancement on a derived copy. Automatic noise reduction
  can remove consonants or create musical artifacts that a voice model learns.
* VOR is a storage/battery convenience, not a voice gate. If it is enabled,
  keep a short pre-roll and post-roll if the device supports them; otherwise
  initial/final phonemes can be clipped. A continuous file is preferable for
  diarization because turn boundaries can be recomputed.
* A 68-hour battery claim does not mean one file can be safely recorded for 68
  hours. Start a fresh file at least daily (or at a natural break), verify that
  the device did not fill the card, and copy files with checksums.

On macOS, WAV, AIFF, CAF, AAC/ALAC, and MP3 are supported Core Audio file/data
combinations. Apple documents `afconvert` for deterministic conversion and
linear PCM packaging in [Using Audio](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/MultimediaPG/UsingAudio/UsingAudio.html)
and the [Core Audio format table](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/CoreAudioOverview/SupportedAudioFormatsMacOSX/SupportedAudioFormatsMacOSX.html).
For this project, preserve the recorder WAV and make a normalized 16 kHz mono
PCM working copy for ASR/diarization; never overwrite the source.

## Three different operations (do not conflate them)

| Operation | Question answered | Output | What it cannot do |
| --- | --- | --- | --- |
| VAD / speech activity detection | “Is there speech here?” | speech/non-speech spans | Cannot tell whose speech it is. |
| Speaker diarization | “Which anonymous speaker is active at each time?” | `speaker_0`, `speaker_1`, … time spans (and overlap labels when supported) | Labels are anonymous until matched to an enrolled voice; diarization does not magically unmix overlapping waveforms. |
| Source separation / target-speaker extraction | “Can a model estimate one voice waveform from the mixture?” | a target stem plus residual/other stems | It can introduce artifacts and fail when voices overlap heavily or sound alike. The result still needs speaker verification. |

Use all three in that order. A common mistake is to run Whisper on the whole
file and assume that a transcript's speaker labels are clean training audio;
ASR timestamps do not remove another person speaking at the same time.

## Local-first model choices on an M1 Mac

### Recommended default: MLX VAD + MLX diarization + local speaker verification

`mlx-audio` is an MLX library for Apple Silicon and currently documents a
lightweight Silero VAD and NVIDIA Sortformer diarization models (up to four
speakers, including a streaming variant). Its README also documents MLX STT
and TTS integrations. See the [official mlx-audio README](https://github.com/Blaizzy/mlx-audio#voice-activity-detection--speaker-diarization-vad).

Sortformer supplies time segments and anonymous speaker IDs. To decide which ID
is the owner, enroll with several clean clips recorded on the same device and
in a few normal speaking styles. Compute a speaker embedding for the enrollment
set and compare each diarized segment with cosine similarity. SpeechBrain's
official VoxCeleb recipe documents the pretrained ECAPA-TDNN speaker-recognition
model and cosine-similarity verification workflow in its
[Speaker Recognition README](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/README.md).
SpeechBrain is PyTorch rather than MLX, so it is a **small local fallback** on
MPS/CPU, not the first MLX path. Cache the enrollment embedding and calibrate a
threshold from a small labeled sample of the owner's and other speakers'
segments; do not copy a generic threshold into production.

The app should write a decision record per segment:

```json
{
  "segment_id": "file-2026-07-30T10:15:02.400Z-000123",
  "start_s": 902.4,
  "end_s": 907.8,
  "diarized_speaker": "speaker_1",
  "owner_similarity": 0.82,
  "overlap": false,
  "decision": "accept",
  "reason": "owner_threshold_passed"
}
```

Use three buckets, not a binary delete:

* `accept`: owner similarity is above the calibrated threshold and no overlap
  flag is present;
* `review`: low margin, short segment, clipping, or disagreement between
  embedding windows; and
* `reject`: a different enrolled speaker or non-speech.

Any overlap flag or any non-target speaker inside the candidate clip is an
immediate `reject`, not a `review`. Only `accept` is eligible for automatic
dataset construction. Preserve links to rejected raw spans for audit, but do
not provide a promotion path for mixed-speaker clips.

### pyannote.audio fallback (strong diarization, not MLX)

The open-source `pyannote.audio` toolkit provides pretrained local diarization
pipelines, speaker embeddings, and overlap-aware processing. Its official
repository documents the `community-1` pipeline and local execution in
[pyannote-audio](https://github.com/pyannote/pyannote-audio). It is PyTorch-based;
use it when MLX Sortformer is not accurate enough or when a known pipeline is
needed for comparison. Model downloads require a Hugging Face token and model
terms; download once, pin an exact revision, cache it, and process files
offline afterward.

The pyannote release notes also describe its optional speech-separation
pipeline, but separation is a separate capability from diarization. It can be
used for difficult overlaps on a machine with adequate memory, then checked
with the owner embedding. It should not replace the default diarization-plus-
verification path on an M1.

### Overlap handling: reject, do not rescue

Source-separation systems such as Meta's [SAM-Audio repository](https://github.com/facebookresearch/sam-audio)
may estimate a target stem, but they can introduce artifacts and cannot
guarantee that another speaker's phonemes were removed. They are explicitly out
of the dataset path. The app records the overlap reason and rejects the whole
candidate clip.

## Enrollment protocol

The enrollment clips are not the training corpus. They are a reference used to
answer “which diarized cluster sounds like me?” Record at least several clean
clips (roughly 30–60 seconds total as a starting point; 2–5 minutes across
different distances and normal styles is safer for an all-day recorder). Include
the same microphone, room types, volume range, and neutral/warm/energetic
styles expected in the source files. Keep a held-out clip that is never used to
set the threshold.

The app should let the user listen to a contact sheet of accepted/review/reject
segments before any training run. If the owner changes microphones or the
recorder's noise-reduction setting, create a new enrollment set and record that
fact in the manifest.

## End-to-end procedure for an all-day file

1. **Copy and checksum.** Copy from the recorder's USB disk into an immutable
   intake directory. Compute SHA-256 before processing. Never process directly
   from a removable disk.
2. **Inspect.** Read the actual container/codec, sample rate, channels, bit
   depth, duration, and clipping. Fail closed on unsupported or corrupted audio.
3. **Normalize a derivative.** Decode to mono 16 kHz PCM for VAD, diarization,
   embeddings, and Whisper MLX. Retain the original-rate source for accepted
   training clips. Use bounded chunks (for example 5–10 minutes) with a small
   overlap so a boundary cannot lose a word; stitch spans by source timestamps.
4. **VAD.** Remove long non-speech spans and keep a small context collar. Do not
   trim tightly before diarization; breath and plosive context helps identity
   checks.
5. **Diarize.** Run MLX Sortformer (or the pinned pyannote fallback), requesting
   the expected maximum number of speakers where supported. Save model revision,
   windowing parameters, and diarization output.
6. **Identify the owner.** Score each turn against the enrollment embedding;
   calibrate and apply accept/review/reject policy. If the candidate clip has
   an overlap flag or any non-target speaker, reject the entire clip.
7. **Transcribe accepted spans.** Run the already-installed Whisper MLX/large-v3
   (or mlx-audio Whisper) on accepted clips, retaining word timestamps and the
   original source offsets. Correct the transcript before using it as reference
   text for TTS training.
8. **Human gate.** Listen to a random sample and every low-confidence decision.
   Rejected mixed-speaker clips remain auditable but cannot be promoted.
9. **Build a versioned dataset.** Store only accepted audio plus manifest,
    checksums, source offsets, speaker score, processing-model revisions, and
    licensing/privacy metadata. The raw intake remains immutable and separate.

For a very long recording, process one file at a time and checkpoint after each
chunk. A crash should resume from the last completed chunk rather than restart
or redownload any model. This also follows the workspace rule to never burst
local bus requests, model downloads, or external services.

## Where it will fail (and how the app should respond)

* **Overlapping speech:** diarization can mark overlap, but the waveform is a
  mixture. Reject the entire candidate clip. Separation is not used to rescue
  training data.
* **Very short turns:** embeddings are unstable on a few hundred milliseconds.
  Merge adjacent same-speaker turns when the gap is below the configured collar,
  or send the span to review.
* **Similar voices / changed acoustics:** identity scores shift with distance,
  room, microphone orientation, and noise reduction. Calibrate per recorder
  and keep held-out validation clips.
* **Clipping and VOR cuts:** clipping cannot be repaired reliably. Reject spans
  with excessive clipping; retain pre/post-roll when available.
* **Background TV/radio or playback of the owner's voice:** a speaker model may
  accept the recording as “you.” Treat playback as a separate source and
  avoid using it for training; the human gate is required.
* **Privacy/consent:** an all-day recorder may capture other people. Keep the
  data local, provide a delete path for raw and derived files, and follow the
  applicable consent law before retaining or training on other people's speech.

## Decision

Implement the first version as **MLX-first VAD → MLX Sortformer diarization →
local ECAPA speaker verification → hard overlap rejection → human review**, with
Whisper MLX after filtering. Add pyannote only as a diarization comparison
provider; do not add an overlap-rescue path to the training dataset. This gives
the user the requested all-day capture workflow without claiming that
inexpensive hardware can perfectly unmix simultaneous conversations.
