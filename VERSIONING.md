# Versioning and change-tree policy

VoiceCloneGPT uses Semantic Versioning for software and an independent version
for each trained voice-model bundle. The app is a private utility, but strict
versioning is required because recordings, manifests, model checkpoints, and
the local data-bus protocol must remain reproducible.

## Software versions

`MAJOR.MINOR.PATCH` starts at `0.1.0` while the app is being built.

### PATCH (`0.1.x`)

Use for a backward-compatible correction that does not change a public seam:

- bug fixes;
- test-only changes;
- documentation and ICM contract clarifications;
- dependency/security updates with unchanged interfaces;
- performance or UI polish with unchanged behavior.

### MINOR (`0.x.0`)

Use for a backward-compatible capability:

- a new importer, style, provider, screen, or optional bus message;
- a new module with an existing interface;
- additive manifest/model metadata fields;
- a new app feature that existing projects can ignore.

### MAJOR (`x.0.0`)

Use for any incompatible change:

- changing or removing a required input/output or public module seam;
- breaking a data-bus message or model-bundle schema;
- changing the meaning or units of an existing field;
- moving/deleting a stage in a way that invalidates existing artifacts;
- replacing the runtime, model format, or app protocol without an adapter;
- changing the on-disk layout so existing runs cannot resume.

Every major change requires a migration note, an explicit compatibility decision,
and a new major branch/tag. Never silently rewrite a prior major line.

## Branch and tag tree

- `main` is the integration root and always points at the latest accepted version.
- `feature/v0.2.0-<short-name>` is for a planned minor capability.
- `fix/v0.1.1-<short-name>` is for a patch correction.
- `breaking/v1.0.0-<short-name>` is for a major migration.
- Accepted releases are tagged `vMAJOR.MINOR.PATCH`.

Workers must not commit directly to `main` while a versioned task branch exists.
Every task reports its intended version impact before editing. The orchestrator
chooses the branch and release tag after reviewing the worker's change surface.

## Voice-model bundle versions

Software version and voice-model version are separate. A model bundle records:

- `model_version: MAJOR.MINOR.PATCH`;
- `base_model` and exact revision;
- dataset manifest/checksum;
- training configuration and checkpoint;
- runtime format (`pytorch`, `mlx`, `onnx`, etc.);
- license metadata;
- acceptance-test results.

Retraining with the same bundle schema is a model PATCH. Adding a compatible
style or metadata field is a model MINOR. Changing the loader-required files or
tensor contract is a model MAJOR and requires a migration or adapter.

## Worker completion gate

Before reporting completion, every coding worker must:

1. Read this file and state the version impact.
2. Keep the task on the correct branch/worktree.
3. Run `/icm-check`, apply every actionable recommendation, and rerun it.
4. Run the relevant tests/build.
5. Record the version, migration note (if any), and verification evidence.

6. Obtain the independent cross-agent review required by
   [`CROSS_REVIEW.md`](CROSS_REVIEW.md). A change is not merge-ready until the
   routed reviewer approves it or the orchestrator records a documented,
   user-approved exception.
