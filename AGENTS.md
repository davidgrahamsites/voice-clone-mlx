# VoiceCloneGPT agent routing

VoiceCloneGPT is a local, personal, non-commercial voice workspace with two
independently deletable apps and shared versioned artifact contracts.

- Current priorities and app behavior: [`PLAN.md`](PLAN.md)
- Workspace pipeline and task routing: [`CONTEXT.md`](CONTEXT.md)
- Version hierarchy, branch tree, and worker release gate:
  [`VERSIONING.md`](VERSIONING.md)
- External API/service safety gate: [`OPERATIONS.md`](OPERATIONS.md)
- Plain-language reporting gate: [`REPORTING.md`](REPORTING.md)
- Optional high-risk review checklist: [`CROSS_REVIEW.md`](CROSS_REVIEW.md)
- Voice-model producer/consumer contract:
  [`docs/architecture/voice-model-lifecycle.md`](docs/architecture/voice-model-lifecycle.md)
- Current TTS backend decision:
  [`docs/research/tts-backend-evaluation.md`](docs/research/tts-backend-evaluation.md)

Read the contract for the module being changed and only its named inputs. Keep
UI, domain workflow, providers, and filesystem artifacts behind separate seams.
Run relevant tests after every coding task. Run `/icm-check` after a substantial
feature, integration, packaging, release, architectural refactor, or file move
that changes module boundaries. Small targeted fixes do not require a separate
ICM pass when they remain inside an already-audited seam.

All status updates, worker handoffs, error reports, and final reports follow
[`REPORTING.md`](REPORTING.md). Use plain talk and define jargon inline on first
use.

Independent cross-family review is not a merge requirement. The orchestrator
may use [`CROSS_REVIEW.md`](CROSS_REVIEW.md) for security-sensitive, destructive,
external-service, model-format, or other high-risk changes, or when the user
explicitly requests it. Ordinary changes advance through tests and the
milestone ICM gate.

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
reports changed seams, chosen version level, migration impact, test/build
evidence, and the milestone `/icm-check` result when one applies. No worker may silently rewrite
an earlier major line or use a version number merely to label an experiment.

## External-service gate

Before using an API, service, hosted GPU, model registry, browser flow, or local
bus, every worker must read [`OPERATIONS.md`](OPERATIONS.md). The default is
one-at-a-time, rate-limited, cached, checkpointed work with bounded retries.
Bursting, unbounded fan-out, tight polling, and infinite retry loops are
prohibited.

Git operations follow the same rule: never burst Git commands across workers,
terminals, or retry loops.

## Read-only worker default

Workers are **read-only by default**. A worker may inspect files, search,
research, run bounded non-mutating checks, and prepare a proposed diff, but it
must not create or edit files, switch branches, commit, push, install
dependencies, call mutating APIs, upload data, or change an external service.

Before any write, the worker must send the orchestrator a write request naming:
the exact paths or service operation, reason, intended version/branch, data
impact, test plan, whether the milestone `/icm-check` gate applies, and
rollback. The orchestrator must reply
with an explicit `APPROVED WRITE` naming the same scope. Silence, a general task
assignment, or a previous approval does not authorize a new write. If the
scope changes, the worker must request approval again.
