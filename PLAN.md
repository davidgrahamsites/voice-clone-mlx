# VoiceCloneMLX plan

VoiceCloneMLX consists of two local-first applications that share a versioned
voice model and a normalized text format.

## App 1: Voice Studio

Voice Studio creates and maintains the personal voice model.

1. Import the user's existing `.txt`, `.docx`, `.pdf`, and authorized website
   material as candidate writing samples for a recording script.
2. Generate a recording script that preserves the user's vocabulary and subject
   matter while adding missing phonetic, punctuation, numerical, and expressive
   coverage. Deduplicate passages and break them into recordable utterances.
3. Organize the script into manageable sessions and style blocks. Include a
   spoken style marker, a silent pause, calibration text, and the reading
   passages for every block.
4. Review and edit the generated script in a teleprompter-friendly view before
   recording.
5. Record or upload one continuous lossless audio session.
6. Recognize spoken section markers such as:
   - "This is the neutral reading."
   - "This is the warm reading."
   - "This is the energetic reading."
   - "This is the serious reading."
   - "This is the somber reading."
7. Accept common marker variants, including `netural`, but normalize them to
   the canonical style name `neutral`.
8. Use the user's locally installed MLX Whisper resources for private,
   Apple-Silicon-native transcription and timestamp extraction. The preferred
   integration target is the bundled runtime at
   `/Users/appleadmin/Apps/WhisperFlow/WhisperFlow V2.app/Contents/Resources/mlx_whisper/`;
   the WhisperFlow app/environment paths remain configurable. Combine its
   timestamps with the known script for alignment; keep a separate forced
   aligner optional if word-boundary precision later proves insufficient.
9. Remove the marker speech, split the remaining recording into sentence-level
   WAV clips, and route each clip to `data/dataset/<style>/audio/`.
10. Pair every clip with exact text, timestamps, recording-session provenance,
   and style metadata.
11. Present uncertain boundaries and transcription mismatches for human review.
12. Create train, validation, and untouched test manifests split by recording
   session.
13. Fine-tune the selected backend (Qwen3-TTS is the leading candidate, not a
    final decision) and retain versioned checkpoints or adapters, settings,
    evaluations, and reference clips. A reference-audio clone is a baseline;
    the deliverable is an immutable `fine_tuned_full` or `fine_tuned_adapter`
    model bundle that Voice Reader can load without Voice Studio.

The recording conditions remain technically consistent while vocal delivery
varies deliberately. The initial target distribution is 60–70% natural neutral
or narrative speech, with the remainder distributed across warm, energetic,
serious, somber, questioning, emphasis, and dialogue material.

## App 2: Voice Reader

Voice Reader turns owned source material into audio using a selected version of
the personal voice.

1. Import `.txt`, `.docx`, text-based `.pdf`, or a website URL.
2. Extract primary content and preserve useful structure such as titles,
   headings, paragraphs, lists, dialogue, and chapter boundaries.
3. Exclude website navigation, advertisements, cookie notices, and repeated
   page furniture. Require an explicit review for scanned PDFs or uncertain
   extraction.
4. Normalize typography, numbers, dates, abbreviations, URLs, and pronunciation
   overrides into a reviewable project document.
5. Split text at sentence and paragraph boundaries for stable generation.
6. Let the user select any installed voice bundle (the user's voice or another
   consented voice), then assign a default style and optional styles to individual
   passages before synthesis.
7. Generate resumably, cache completed chunks, preview/regenerate individual
   passages, and export WAV or MP3 plus a generation manifest.

Only user-owned or authorized material should be processed. Website ingestion
must respect authentication, access controls, and applicable site terms.

## Shared pipeline

```text
source material -> normalized document -> reviewed segments
                                      -> voice + style selection
                                      -> generated audio chunks
                                      -> assembled WAV/MP3

recording + script -> MLX Whisper + script alignment -> reviewed styled dataset
                                                     -> fine-tuned voice versions
```

## Initial public seams

- Voice Studio accepts a recording and script manifest and emits a reviewable,
  styled dataset.
- Voice Reader accepts a supported source and emits a reviewable normalized
  document; synthesis then emits audio chunks and a final audio file.
- Shared artifacts use documented manifests so either app can be replaced
  without invalidating the recordings, extracted documents, or trained voice.
