---
type: module-contract
module: synthesis-runtime-registry
sequence: semantic
version_impact: minor
---

# Synthesis runtime registry

One job: give a Reader composition root an explicit mapping from a runtime id
to a provider adapter. It never chooses a fallback, downloads a model, loads
weights, or owns a user interface.

## Inputs

- A non-empty runtime id.
- An object implementing `load(artifact_path, manifest)` and
  `synthesize(model, text)`.

## Outputs

- The exact registered adapter, or a typed `RuntimeRegistrationError`.
- `NullRuntime` produces a deterministic silent WAV for local contract and app
  packaging tests. It is not a voice model and must never be presented as one.

## Boundaries

The registry is provider-neutral. MLX, Qwen, and F5 adapters register here as
separate leaf modules when their runtime sources are available. No adapter may
download, retry, or silently substitute another runtime.

## Human check

Register two ids, select each explicitly, and confirm an unknown id fails before
model initialization. Delete this folder and the shared bundle reader and
round-trip contract must still import and test.
