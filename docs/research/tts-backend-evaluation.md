---
type: research-contract
contract: personal-voice-tts-backend-evaluation
status: decision-ready
checked_at: 2026-07-30
inputs:
  - ../../PLAN.md
outputs:
  - private-use backend ranking
  - permissive-reuse backend ranking
human_check: Select the primary backend before implementing a provider adapter.
---

# Personal voice TTS backend evaluation

One job: select a high-quality personal-voice backend for an M1 Mac while
keeping the model bundle and Reader loader provider-neutral.

## Decision

Use **Qwen3-TTS 12Hz 0.6B Base** as the first implementation target and keep
**F5-TTS v1** as the private-use comparison backend.

Qwen is the best overall fit because its official Base model supports rapid
reference-voice cloning and supervised fine-tuning, its code and official
weights are Apache-2.0, and MLX-Audio provides a maintained community inference,
conversion, streaming, and local-serving path. Fine-tuning remains an isolated
remote-CUDA stage; the promoted source checkpoint is preserved and an MLX
conversion is a separately verified runtime variant.

F5 remains attractive for private experimentation: it has expressive
reference-audio cloning, an official fine-tuning workflow, Apple-Silicon
PyTorch instructions, and a community MLX port. Its official pretrained weights
are CC-BY-NC-4.0, and an official-weight fine-tune remains a non-commercial
derivative. That is acceptable for this private utility but prevents F5 from
winning the future permissive-reuse track.

This is an engineering recommendation, not legal advice.

## Hard constraints and evidence rules

- Prefer MLX for transcription, alignment, TTS inference, local serving,
  conversion, and automated evaluation when a maintained implementation exists.
- Use ordinary CPU audio tools for decoding, trimming, resampling, checksums,
  and manifest generation; moving those jobs to MLX adds no useful acceleration.
- Use remote CUDA only for training/fine-tuning without a reliable MLX recipe.
- Do not infer recording-hour requirements from zero-shot cloning capability.
  Official recipes generally do not publish data-duration quality curves.
- Keep code license, pretrained-weight license, training-data provenance,
  fine-tuned derivative terms, and output-use terms as separate metadata fields.

## Ranked recommendations

### A. Best for this private, local utility

1. **Qwen3-TTS 0.6B Base** — primary implementation target. Strong personal
   cloning and official fine-tuning, Apache-2.0 code/weights, and the strongest
   current M1 route through community MLX-Audio. The 0.6B size is preferred for
   first M1 validation; 1.7B remains a quality benchmark.
2. **F5-TTS v1** — private-use comparison backend. Strong expressive cloning,
   official fine-tuning, and a community MLX port, but official weights and
   their fine-tuned derivatives are non-commercial.
3. **Chatterbox Multilingual** — expressive zero-shot comparator with MIT code
   and weights, explicit exaggeration/control parameters, MPS upstream support,
   and community MLX inference. No official personal-speaker fine-tuning recipe.
4. **CosyVoice 3** — feature-rich zero-shot and instruction-driven comparator,
   but CUDA-centric with no established MLX path and no documented personal-data
   minimum.
5. **OpenVoice V2** — permissive, lightweight zero-shot voice conversion with
   granular style controls; no established MLX or official personal fine-tune.
6. **StyleTTS2** — strong published expressiveness and the only concrete
   one-hour fine-tuning example, but its fragile training path, incomplete
   multi-speaker fine-tuning, dependencies, and lack of MLX make it a poor M1
   product seam.
7. **Kokoro-82M** — excellent compact MLX/CPU fallback for fixed voices, but its
   official model card says it is not a voice-cloning model.
8. **Piper** — efficient CPU/ONNX fallback and trainable from owned data, but
   less expressive, not zero-shot, and the maintained engine is GPL-3.0.

### B. Best for future permissive reuse

1. **Qwen3-TTS Base** — best combination of personal cloning, official
   fine-tuning, Apache-2.0 code/weights, and quality potential. Public training
   data provenance is not detailed enough to claim a clean-room lineage.
2. **Chatterbox Multilingual** — MIT-labeled voice-cloning model with strong
   expressive controls; published training-data provenance remains broad.
3. **OpenVoice V2** — MIT-labeled zero-shot converter and a credible lighter
   provider seam; no official fine-tuning route.
4. **Kokoro-82M** — strongest compact Apache-2.0 fixed-voice fallback, not a
   solution for adapting the user's voice.

Exclude official F5 weights and their fine-tuned derivatives from this second
ranking. Do not describe any candidate as fully unrestricted without reviewing
the exact selected revision, model card, notices, and dataset lineage.

## Recording-time decision matrix

| Clean user audio | Supported plan | Evidence-bounded expectation |
|---|---|---|
| 20–30 minutes | Curate multiple clean reference clips across neutral and expressive styles; benchmark Qwen, F5, and Chatterbox zero-shot; optionally run an experimental Qwen/F5 fine-tune | Official sources establish cloning from seconds-long references, not a guaranteed fine-tune quality level at 20–30 minutes. This is the preferred first milestone because it minimizes recording burden. |
| 1–2 hours | Expand phonetic/style coverage and run controlled Qwen/F5 fine-tune experiments against the frozen zero-shot baseline | Qwen and F5 publish recipes but no duration/quality threshold. StyleTTS2 reports a concrete one-hour example that is slightly worse than its 24-hour from-scratch model, but requires heavy CUDA resources and is not the recommended backend. |
| 5+ hours | Collect only if acceptance tests show specific deficits; train/evaluate additional checkpoints without changing the bundle contract | No reviewed official Qwen/F5 source establishes five hours as necessary. More data may improve coverage, but promotion is governed by evaluation, not elapsed recording time. |

The app must never tell the user that eight to ten hours is required. Recording
continues only when a failed acceptance dimension identifies missing material.

## Stage placement and unavoidable non-MLX boundaries

| Stage | Preferred implementation | Boundary |
|---|---|---|
| Transcription/timestamps | Apple `mlx-whisper` using the configurable WhisperFlow resources | Official MLX implementation; outputs a versioned transcription manifest. |
| Script alignment | MLX-Audio Qwen forced aligner where validated; otherwise script-aware CPU alignment | Community MLX component behind an alignment adapter. |
| Decode/resample/segment | CPU tools such as ffmpeg/soundfile | Deterministic audio utility module; no model dependency. |
| Qwen inference/generation | MLX-Audio Qwen3-TTS 0.6B | Community MLX adapter behind the provider-neutral synthesis interface. |
| F5 inference/generation | `f5-tts-mlx` for comparison; official PyTorch/MPS as parity reference | Community MLX adapter; never imported by shared contracts. |
| Local serving | MLX-Audio loopback-only server or in-process adapter | Reader depends on a stable loader/synthesizer port, not the server implementation. |
| Fine-tuning | Official Qwen or F5 recipe on remote CUDA | Training-run output is never loaded directly by Reader. |
| Model conversion | `mlx_audio.convert` for supported Qwen checkpoints | MLX artifact is a runtime variant, not a replacement for the promoted source checkpoint. |
| Evaluation | Local MLX generation + MLX Whisper/alignment metrics + human listening gate | Backend-neutral report binds source and runtime-variant checksums. |

## Implementation implications

1. `VoiceModelBundle` remains provider-neutral. It identifies a source artifact
   plus zero or more runtime variants, each with backend id, format, converter,
   quantization, revision, checksum, and license metadata.
2. `VoiceModelLoading` verifies the bundle, detects M1 capabilities, and asks a
   backend registry for a compatible runtime. It does not convert, download,
   train, or synthesize.
3. A `qwen3_mlx_inference` leaf adapter is the first code slice. An
   `f5_mlx_inference` leaf adapter may be added without editing Reader or shared
   contracts.
4. Remote training emits an append-only training-run manifest and source
   checkpoint. Promotion requires acceptance evidence. Conversion happens after
   promotion and must pass parity tests before its runtime variant is selectable.
5. The first data milestone is 20–30 minutes plus a curated reference library.
   A 1–2 hour collection is triggered only by measured quality gaps.

## Acceptance contract

Before a backend/runtime variant can be selected:

- the bundle schema, every artifact checksum, source revision, converter
  provenance, and all license layers validate;
- Reader can load a fixture bundle without importing Voice Studio or training
  code;
- Studio can promote a bundle without importing Reader;
- deleting the Qwen or F5 adapter leaves shared contracts and the other adapter
  tests buildable;
- the selected MLX runtime generates valid WAV output on the target M1 and the
  declared sample rate/channel contract matches the file;
- source-runtime and MLX-runtime outputs pass declared intelligibility,
  alignment, duration, and human-listening parity gates;
- capability selection never silently substitutes CPU, MPS, CUDA, or another
  provider; the chosen device/runtime appears in the generation manifest;
- an unavailable or incompatible MLX variant produces a typed error or an
  explicitly approved fallback;
- the 20–30 minute evaluation is compared with the same frozen prompts and
  references as later fine-tuned checkpoints;
- F5 bundles preserve MIT code and CC-BY-NC weight/derivative notices and a
  `personal-noncommercial` usage statement.

## Primary sources

- [Qwen3-TTS repository and official fine-tuning recipe](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 12Hz 0.6B Base model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [Qwen3-TTS technical report](https://arxiv.org/abs/2601.15621)
- [F5-TTS repository, license statement, training, and Apple-Silicon instructions](https://github.com/SWivid/F5-TTS)
- [F5-TTS paper](https://arxiv.org/abs/2410.06885)
- [F5 official-weight derivative licensing clarification](https://github.com/SWivid/F5-TTS/discussions/997)
- [Chatterbox repository](https://github.com/resemble-ai/chatterbox)
- [Chatterbox official model card](https://huggingface.co/ResembleAI/chatterbox)
- [Kokoro-82M official model card](https://huggingface.co/hexgrad/Kokoro-82M)
- [OpenVoice repository](https://github.com/myshell-ai/OpenVoice)
- [StyleTTS2 repository and one-hour fine-tuning recipe](https://github.com/yl4579/StyleTTS2)
- [Piper legacy training guide](https://github.com/rhasspy/piper/blob/master/TRAINING.md)
- [Maintained OHF Piper repository](https://github.com/OHF-Voice/piper1-gpl)
- [CosyVoice repository](https://github.com/QwenAudio/CosyVoice)
- [Apple MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [MLX-Audio models, conversion, and local server](https://github.com/Blaizzy/mlx-audio)
- [Community F5-TTS MLX port](https://github.com/lucasnewman/f5-tts-mlx)

