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

## Local capability detection

`local_capability_detector.py` probes the OS and installed packages and reports
what execution capabilities are available. `detect_capabilities()` returns a
frozen `LocalCapabilitySet` with one field per capability: `has_pytorch`,
`has_mlx`, `has_cuda`, `has_metal`, `python_version`, `torch_version`,
`mlx_version`. Probes are pure import spec checks (no initialization, no API calls).
Hardware flags are conservative (false unless verified). Versions come from
package metadata, not by importing. `__post_init__` validates all field types,
consistency (cuda/metal require pytorch, versions require presence), and rejects
invalid combinations.

Pure: no filesystem, network, subprocess, or clock imports. No top-level torch
or mlx import. Stdlib only: `importlib.util`, `importlib.metadata`, `platform`.

TDD evidence (measured at this worktree):

    focused: PYTHONPATH=src python3 -m pytest \
      tests/voice_reader/test_local_capability_detector.py -q
      20 passed
    baseline (test ignored): 1989 passed, 9 skipped
    full suite:              2009 passed, 9 skipped
