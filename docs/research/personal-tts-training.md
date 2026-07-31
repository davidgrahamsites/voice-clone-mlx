# Personal English TTS training at free or near-zero cost (July 2026)

## Recommendation in one paragraph

Use **Qwen3-TTS 12 Hz 0.6B Base** first for local zero-shot cloning, then single-speaker fine-tune it on a carefully recorded corpus if blind listening tests show a worthwhile gain. It is the best starting point here because the current official release supports 3-second cloning, local inference, and official single-speaker fine-tuning; its repository and released weights are Apache 2.0. The 0.6B model is much more plausible on a free 16 GB GPU than the 1.7B model. Record the full 8–10 hours if the goal is a durable voice asset, but do not train a foundation model from scratch: fine-tune the pretrained base. Keep **Piper** as a second, separately trained deployment target if instant CPU inference and a small ONNX file matter more than maximum naturalness.

This is a technical reading of the published licenses, not legal advice.

## The three different jobs people call “training”

1. **Zero-shot voice cloning:** no weights change. At synthesis time the model receives a short reference clip and imitates its speaker identity/style. This is the free, immediate baseline and may already be sufficient.
2. **Fine-tuning:** start with a broadly trained base model and update weights using the target speaker's transcribed audio. This is the sensible use of 10 hours and can improve consistency, pronunciation, cadence, and speaker identity.
3. **Training from scratch:** initialize a TTS system without pretrained TTS weights. Ten hours is enough for some older single-speaker architectures, but not enough to recreate a modern general-purpose foundation model. It costs far more compute, loses the base model's linguistic coverage, and is the wrong “best from the beginning” choice.

The durable investment is the clean, lossless, accurately transcribed dataset. Models change quickly; that corpus can be reused.

## Model comparison

| Model | What is officially available | License issue | Fit for this project |
|---|---|---|---|
| **Qwen3-TTS 12 Hz Base (0.6B/1.7B)** | 3-second voice cloning, local Python inference, and official single-speaker SFT for both sizes. Training input is WAV + exact text + a shared reference WAV. | Apache 2.0 repo/model release. | **Primary choice.** Start with 0.6B; try 1.7B only with substantially more VRAM. [Official repo](https://github.com/QwenLM/Qwen3-TTS) · [official fine-tuning recipe](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) · [license](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) |
| **Chatterbox** | Excellent local zero-shot cloning from a short reference; expressive controls and built-in watermarking. The official public repo emphasizes inference and does not publish a personal-speaker training recipe. | MIT, including published models according to Resemble. | Best **zero-shot comparison**, not the main fine-tuning path. [Official repo](https://github.com/resemble-ai/chatterbox) · [official product explanation](https://www.resemble.ai/learn/models/chatterbox) |
| **F5-TTS** | Official code includes training/finetuning; flow-matching synthesis can be very natural. | Code is MIT, but official pretrained weights are CC BY-NC because of Emilia; a fine-tune remains noncommercial. Training a clean base from scratch avoids those weights but defeats the low-cost goal. | Good research alternative for strictly personal/noncommercial use, weaker licensing choice for a long-lived asset. [Official repo](https://github.com/SWivid/F5-TTS) · [license clarification](https://github.com/SWivid/F5-TTS/discussions/997) |
| **Fish Speech** | Powerful speech-language-model family with training support. | The current Fish Audio Research License permits free research/noncommercial use; commercial use, including derivatives/fine-tunes, needs a separate agreement. | Skip unless personal-only use is certain and its license is acceptable. [Official repo](https://github.com/fishaudio/fish-speech) · [license](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) |
| **XTTS v2 / Coqui TTS** | Mature zero-shot cloning and a convenient Colab/Gradio fine-tune that trains only the GPT encoder; official docs say the implementation does not train the whole model. | XTTS weights use the Coqui Public Model License rather than the repo's MPL; Coqui shut down and the upstream is effectively legacy. | Still usable, but no longer the best greenfield choice. [Official XTTS docs](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md) · [model license](https://coqui.ai/cpml) |
| **StyleTTS2** | High-quality English single-speaker training recipe, but the official repo still lists multispeaker testing/fine-tuning work as unfinished and notes second-stage DDP problems. | Code is MIT; pretrained-model usage has additional voice-consent/disclosure terms and inference may depend on a GPL phonemizer package. | Capable but fragile and operationally expensive; not the first choice for a one-person project. [Official repo](https://github.com/yl4579/StyleTTS2) · [paper](https://arxiv.org/abs/2306.07691) |
| **Piper (current OHF fork)** | Straightforward single-speaker training/fine-tuning and ONNX export; CPU-oriented local inference. Official docs report successful training with 8 GB VRAM, while maintainers used 24–48 GB. | Current maintained `piper1-gpl` code is GPL-3.0; individual checkpoint/data licenses must also be checked. | Best lightweight/offline target; likely less expressive than Qwen3-TTS. [Official maintained repo](https://github.com/OHF-Voice/piper1-gpl) · [training guide](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md) |
| **OpenVoice v2** | Instant tone-color conversion plus style control; it converts a base speaker's generated speech into the target timbre rather than learning a bespoke personal TTS model. | MIT. | Useful experiment or conversion layer, but not the requested 10-hour fine-tuned narrator. [Official repo](https://github.com/myshell-ai/OpenVoice) · [paper](https://arxiv.org/abs/2312.01479) |

IndexTTS2 is also competitive zero-shot technology, but its custom Bilibili model license has scale thresholds and the public repo does not offer as clean an official personal fine-tuning route. It is not clearly superior for this particular goal. [Official repo](https://github.com/index-tts/index-tts) · [license](https://github.com/index-tts/index-tts/blob/main/LICENSE)

## Recording the 8–10 hour corpus

Quality and consistency matter more than reaching exactly ten hours.

- Record **one sentence per take**, usually 2–12 seconds, in mono lossless WAV. Keep original masters at 24-bit/48 kHz; create model-specific resampled copies later.
- Use one microphone, mouth distance, room, gain, posture, and signal chain. A quiet, treated closet/room beats an expensive microphone in a reflective room. Disable noise suppression, automatic gain, compression, reverb, and “voice enhancement.”
- Aim for healthy peaks around -12 to -6 dBFS with no clipping. Record 30–60 seconds of room tone each session.
- Transcribe **exactly what was spoken**, including contractions, hesitations that are intentionally retained, and the spoken form of abbreviations. Reject misreads, clicks, bumps, background sounds, clipping, and takes with materially different distance/tone.
- Split by recording session so the validation/test sets reveal session drift. Hold out roughly 5% validation and 5% final test; never tune against the final test set.
- Cover ordinary prose, dialogue, questions, commands, lists, dates, numbers, currency, acronyms, names, punctuation, and difficult English phoneme combinations. Include long-form narration, since that is the intended use.
- For expressive material, label or segregate styles (neutral, warm, excited, concerned, whispered, etc.). Do not randomly perform every sentence: for a reliable narrator, make about 60–70% clean neutral/narrative speech and distribute the rest deliberately among a small set of repeatable styles.
- Take breaks. Vocal fatigue and changing mic geometry create a worse dataset than fewer consistent hours. Record a fixed calibration paragraph at the start and end of every session to detect drift.
- Obtain and retain written consent for every voice in the data. Do not include copyrighted audiobooks or other speakers as unlicensed training material.

Ten hours at an average of six seconds per accepted utterance is roughly 6,000 clips. Budget extra recording time because rejects are normal. Before the full campaign, make a 20–30 minute pilot in the final setup and inspect waveforms, noise, transcripts, and zero-shot references; this validates the recording chain without adopting a “progressive model” strategy.

## Compute and time: realistic expectations

Official Qwen fine-tuning defaults show batch size 32, while its one-click example uses batch size 2, but the project does not publish a VRAM/time table. Therefore the following are planning estimates, not vendor promises:

- **Qwen3-TTS 0.6B inference:** target an NVIDIA CUDA GPU with roughly 8–12 GB VRAM; lower-memory configurations may require attention/memory optimizations. CPU inference is possible in principle but is unlikely to be pleasant for long-form generation.
- **0.6B full fine-tune:** target 16 GB VRAM with batch size 1–2, gradient accumulation, BF16/FP16, FlashAttention 2, and gradient checkpointing if supported. Ten hours for several epochs is likely a multi-session job—roughly many hours to a few days depending on GPU and settings.
- **1.7B fine-tune:** plan for 24–48 GB VRAM; a free T4/P100 is not a comfortable target. Its quality gain should be proven by zero-shot tests before paying for compute.
- **Piper fine-tune:** documented as workable with as little as 8 GB VRAM; 10 hours is very plausible on free notebook GPUs. Inference exports to a small ONNX model and runs well on CPU.
- **Training a modern base from scratch:** not realistic on free notebook quotas. StyleTTS2's own repository records multi-GPU complications; F5/Qwen foundation training requires far more data and compute than this personal corpus.

Make every job resumable and checkpoint at least once per epoch. Copy checkpoints out of the ephemeral runtime after each session.

## Where to train for $0

1. **Kaggle Notebooks is the best genuinely free first choice.** Kaggle officially provides Tesla P100 GPUs and says the weekly quota is 30 hours or sometimes higher depending on demand. Dataset storage and notebook versions make interrupted jobs manageable. [Kaggle's official GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage)
2. **Google Colab Free is the second choice.** Google says GPU types, quotas, idle timeouts, and availability fluctuate; free notebooks run at most 12 hours and expensive resources are heavily restricted. It is useful for preprocessing, zero-shot evaluation, or short resumable runs, not something to schedule a guaranteed multi-day fine-tune around. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html)
3. **Hugging Face Spaces is for demos, not sustained free training.** Default hardware is CPU. Standard GPUs are billed by the minute; free users can use existing ZeroGPU Spaces for only a small daily quota, and hosting ZeroGPU has account/Space restrictions. A community GPU grant is possible but not guaranteed. [GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus) · [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
4. **Vercel and ordinary shared web hosting are unsuitable.** Vercel Hobby functions have 2 GB RAM/1 vCPU and short duration limits, with no training GPU. Shared-hosting Python processes typically have the same category of limits. Use either only as a thin authenticated frontend that calls a machine running inference. [Vercel function limits](https://vercel.com/docs/functions/limitations)

If free quotas prove too fragmented, the economical fallback is renting one 24 GB or 48 GB GPU for a checkpointed run, not maintaining a server. That is near-zero-cost compared with weeks of engineering around unreliable free sessions.

## Cost-minimizing GPU ladder

Use this order and stop as soon as the acceptance tests pass:

1. **Local M1 preparation:** record, transcribe, reject mixed-speaker clips,
   align, resample, build manifests, and run zero-shot baselines locally. Do
   not spend GPU money before the dataset and prompts are proven.
2. **Free notebook pilot:** run a tiny Qwen 0.6B smoke test and one short
   resumable fine-tune on Kaggle or Colab. Their free GPU availability and
   quotas change, so treat these as opportunistic workers, not guaranteed
   infrastructure. [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage) ·
   [Colab FAQ](https://research.google.com/colaboratory/faq.html)
3. **Preemptible/marketplace rental:** if free sessions cannot finish, rent a
   24 GB consumer GPU first. Vast.ai is a marketplace whose price and host
   reliability vary; RunPod publishes per-second GPU pricing and is usually
   easier to make reproducible. [Vast.ai](https://vast.ai/) ·
   [RunPod pricing](https://docs.runpod.io/serverless/pricing)
4. **48 GB GPU only on evidence:** use an A6000/A40-class machine only if the
   selected training configuration cannot fit on 24 GB. Do not jump to an A100
   or H100 for the first run.

Every paid run must have a preflight estimate, a hard dollar ceiling, an
automatic stop time, checkpoint uploads, and a clean shutdown step. The first
paid job should be a 10–20 minute environment/smoke test; the second should be
one short fine-tune with the 20–30 minute pilot dataset. Only a measured quality
win authorizes the full corpus. Never leave a GPU, disk, or notebook running
between jobs.

The app should make this policy visible in the training manifest:
`provider`, `gpu_type`, `price_source`, `estimated_max_cost`, `hard_cost_cap`,
`deadline`, `checkpoint_uri`, `shutdown_verified`, and `user_approval`. A job
that cannot report those fields is not eligible to start.

## End-to-end execution plan

1. Define acceptance tests before recording: 50 unseen sentences spanning narration, dialogue, numbers/names, questions, emotion, and a 3–5 minute continuous passage. Score intelligibility, speaker similarity, naturalness, unwanted artifacts, and consistency blind against the real recordings.
2. Install Qwen3-TTS locally or in Kaggle and compare 0.6B and 1.7B zero-shot using several clean 5–15 second candidate references. Also test Chatterbox as a zero-shot control. Save audio and generation settings.
3. Lock the recording chain with the pilot, then record and curate the full corpus. Preserve masters, processed WAVs, transcripts, split manifests, consent, microphone/setup notes, and checksums.
4. Fine-tune **Qwen3-TTS 0.6B Base** using the official JSONL/code-extraction/SFT workflow. Use one excellent shared `ref_audio` for all rows, as the official recipe strongly recommends. Begin with batch size 1–2 and 1–3 epochs, evaluate every checkpoint, and stop when held-out quality declines. More epochs are not automatically better.
5. Compare the fine-tuned checkpoint blindly against zero-shot 0.6B, zero-shot 1.7B, and Chatterbox. Promote only if it wins the acceptance set. If identity is good but local speed is poor, train/fine-tune Piper from the same corpus and treat it as a “fast” voice tier.
6. Run inference locally behind a loopback-only Python/Gradio/FastAPI service. Chunk input on sentence boundaries, preserve paragraph pauses, normalize numbers/abbreviations before synthesis, and concatenate with short crossfades. Cache outputs by model version + normalized text + generation settings.
7. Keep remote deployment optional. A subdomain can reverse-proxy to a private GPU machine, but exposing a clone of one's own voice creates impersonation risk. Require authentication, rate limits, logs, HTTPS, and preferably an audible or embedded disclosure/watermark for shared output.

## Bottom line

Recording ten excellent hours is worthwhile, but “best from the beginning” means **building a reusable corpus and fine-tuning a strong pretrained base**, not training from random initialization. In July 2026, the cleanest combination of quality, official training support, local use, and licensing is Qwen3-TTS 0.6B Base. Kaggle can plausibly make the experiment free; budget for a short rented-GPU run only if checkpointed free sessions cannot finish reliably. Piper is the pragmatic CPU deployment companion, not the quality leader.
