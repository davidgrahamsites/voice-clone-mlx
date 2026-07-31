---
type: module-contract
module: shared-model-bundles
sequence: semantic
---

# Shared model-bundle contracts

## Inputs

- Immutable `bundle.json` manifests and runtime payloads under a caller-provided
  voice-model root.

## Process

- Parse provider-neutral bundle metadata.
- Verify schema major version, safe relative paths, and SHA-256 payload checksums.
- List verified bundles and require an exact bundle id for selection.

## Outputs

- `VoiceModelBundle` descriptors only. Model weights are not loaded here.

## Human check

Confirm that a Reader can select two different voice ids by exact bundle id and
that a tampered payload is rejected before backend initialization.
