# VoiceCloneGPT agent routing

VoiceCloneGPT is a local, personal, non-commercial voice workspace with two
independently deletable apps and shared versioned artifact contracts.

- Current priorities and app behavior: [`PLAN.md`](PLAN.md)
- Workspace pipeline and task routing: [`CONTEXT.md`](CONTEXT.md)
- Version hierarchy, branch tree, and worker release gate:
  [`VERSIONING.md`](VERSIONING.md)
- External API/service safety gate: [`OPERATIONS.md`](OPERATIONS.md)
- Voice-model producer/consumer contract:
  [`docs/architecture/voice-model-lifecycle.md`](docs/architecture/voice-model-lifecycle.md)
- Current TTS backend decision:
  [`docs/research/tts-backend-evaluation.md`](docs/research/tts-backend-evaluation.md)

Read the contract for the module being changed and only its named inputs. Keep
UI, domain workflow, providers, and filesystem artifacts behind separate seams.
Run `/icm-check` and relevant tests after every coding task, UI task, refactor,
or file move.

## Required version discipline

Every worker must read [`VERSIONING.md`](VERSIONING.md) before editing. In its
first status update, it must state the intended version impact (`PATCH`,
`MINOR`, or `MAJOR`) and the branch it will use. The worker must not commit to
`main` when a versioned task branch is appropriate.

- A `PATCH` keeps every public seam, schema, artifact layout, and runtime
  contract compatible (`0.1.x`).
- A `MINOR` adds an optional, backward-compatible capability or module
  (`0.x.0`).
- A `MAJOR` changes/removes a required seam, schema, artifact layout, model
  tensor/runtime contract, or stage meaning (`x.0.0`). It requires a migration
  note, compatibility decision, and a `breaking/vX.0.0-<short-name>` branch.

The orchestrator owns branch creation, merge order, and release tags. A worker
reports changed seams, chosen version level, migration impact, `/icm-check`
result, and test/build evidence before handoff. No worker may silently rewrite
an earlier major line or use a version number merely to label an experiment.

## External-service gate

Before using an API, service, hosted GPU, model registry, browser flow, or local
bus, every worker must read [`OPERATIONS.md`](OPERATIONS.md). The default is
one-at-a-time, rate-limited, cached, checkpointed work with bounded retries.
Bursting, unbounded fan-out, tight polling, and infinite retry loops are
prohibited.

## Read-only worker default

Workers are **read-only by default**. A worker may inspect files, search,
research, run bounded non-mutating checks, and prepare a proposed diff, but it
must not create or edit files, switch branches, commit, push, install
dependencies, call mutating APIs, upload data, or change an external service.

Before any write, the worker must send the orchestrator a write request naming:
the exact paths or service operation, reason, intended version/branch, data
impact, tests and `/icm-check` plan, and rollback. The orchestrator must reply
with an explicit `APPROVED WRITE` naming the same scope. Silence, a general task
assignment, or a previous approval does not authorize a new write. If the
scope changes, the worker must request approval again.
