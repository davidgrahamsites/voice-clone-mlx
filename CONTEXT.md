# VoiceCloneGPT workspace

One job: route work between the two app pipelines and their shared artifact
contracts.

## Inputs

- Product direction: [`PLAN.md`](PLAN.md)
- Version and branch policy: [`VERSIONING.md`](VERSIONING.md)
- External-service safety policy: [`OPERATIONS.md`](OPERATIONS.md)
- Plain-language reporting policy: [`REPORTING.md`](REPORTING.md)
- Voice-model lifecycle contract:
  [`docs/architecture/voice-model-lifecycle.md`](docs/architecture/voice-model-lifecycle.md)
- Current backend decision:
  [`docs/research/tts-backend-evaluation.md`](docs/research/tts-backend-evaluation.md)

Do NOT load: all run data, model weights, both app implementations, or prior
runs when one task names a narrower module.

## Process

1. Route recording, alignment, dataset, training, evaluation, and promotion work
   to Voice Studio.
2. Route document normalization, model loading, synthesis, and export work to
   Voice Reader.
3. Route cross-app changes through the provider-neutral versioned contracts.

All workers follow the version and branch policy before touching a seam. A
cross-app or model-bundle contract change is presumed `MAJOR` until the
orchestrator documents why an adapter preserves compatibility. All external
calls follow the workspace safety policy: local-first, bounded, rate-limited,
cached, checkpointed, and cancellable. A worker must stop on rate-limit, quota,
auth, abuse, or repeated-error signals rather than retrying or fanning out.

Workers begin in read-only mode. They may write only after an explicit,
scope-matched `APPROVED WRITE` from the orchestrator; otherwise they return a
proposed diff and stop before changing the workspace or an external service.

All outputs use plain language. Technical terms, abbreviations, and model names
are defined inline before they are reused.

## Outputs

- Voice Studio: reviewed datasets and immutable `VoiceModelBundle` versions.
- Voice Reader: resumable audio chunks and assembled local audio exports.

## Human check

Before crossing an app boundary, open the referenced manifest or contract and
confirm its schema version and checksum match the artifact being handed off.
