# Training cost-control module

One job: decide, before any money is spent, whether one planned paid training
run on a rented remote graphics processor (GPU) is allowed to start.

## Inputs

A `CostManifest` (the written-down plan for the run: provider, GPU type, where
the price came from, quoted price per hour, maximum run time, estimated maximum
cost, hard cost cap, deadline, checkpoint location, whether shutdown was
verified, and whether the user approved it), the price observed right now, and
the current time supplied by the caller.

## Process

`cost_manifest` checks that the plan is complete and self-consistent.
`cost_preflight` recomputes the cost at the observed price and returns an
allow/deny decision listing every blocking reason in plain words. Neither
module contacts a provider, creates a hosted job, starts or stops a machine,
reads a clock, writes a file, or spends money.

## Outputs

A `PreflightDecision` with `approved`, the blocking `reasons`, and the
projected cost in United States dollars.

## Human check

Set the observed price above the planned price and confirm the decision is a
refusal that names the hard cost cap, and that no provider call was made.

## Remote training-run coordinator

`remote_training_run.execute_training(request, trainer)` owns one job: advance
one already-approved training request through exactly one call to an explicitly
injected provider adapter, then return a provider-neutral `TrainingRunManifest`.

### Inputs

- `TrainingRequest`, containing the run identity, accepted dataset evidence,
  backend and base-model provenance, configuration checksum, requested learned
  artifact kind, and any append-only checkpoint history.
- `TrainingProvider`, whose single `train(request, resume_from=...)` method is
  implemented outside this module.

The dataset's declared and independently verified SHA-256 checksums must match.
Both values are supplied by the caller: this module does not read or hash the
dataset. Only `fine_tuned_full` and `fine_tuned_adapter` are accepted;
`reference_clone` cannot satisfy the trained-model path.

### Process

Validate all evidence before crossing the provider boundary. On a resumed run,
pass only the latest checkpoint to the provider while preserving the complete
history. Make exactly one provider call: this coordinator has no retry,
fallback, polling, network, filesystem, conversion, publishing, or app logic.

A provider failure raises `TrainingRunError` with a failed, resumable manifest
and the original exception chained. A successful provider result must be a
nonempty learned `TrainedCheckpoint` whose kind matches the request and whose
checksum is valid. A repeated checkpoint id is refused rather than overwritten.

### Outputs

- Completed `TrainingRunManifest` with the new checkpoint appended; or
- `TrainingRunError.manifest`, preserving prior checkpoints with `status`
  `failed` and a plain failure reason.

The caller owns serialization and checkpoint storage. A later provider leaf
owns remote graphics-processor calls. Promotion, conversion, bundle publication,
and Voice Reader remain separate modules.

### Human check

Inspect the returned manifest before promotion. Confirm its dataset, base-model,
configuration, and checkpoint checksums match the reviewed artifacts, and that
the artifact kind is learned rather than `reference_clone`.

### Version and tests

MINOR (`v0.7.0`): this adds an optional module without changing an existing
schema or seam. No migration is required.

The tests in `tests/voice_studio/test_remote_training_run.py` exercise only the
public interface with local in-memory provider fakes. They prove admission,
checksum binding, one-attempt failure handling, resume history, learned-artifact
validation, and import isolation; they make no provider or network calls.

## Runtime-conversion coordinator

`runtime_conversion.coordinate_mlx_conversion(request, converter)` advances one
human-accepted source-model release to one **pending** MLX runtime candidate.
It is a provider-neutral metadata seam: it never reads or writes model files,
loads a runtime, registers a runtime, runs a subprocess, downloads anything,
or publishes a bundle.

### Inputs

- `ConversionRequest`: accepted source-release identity and checksum, source
  checkpoint path/checksum, provider identity/revision, converter
  identity/revision/status, target format, MLX runtime version, quantization,
  tensor/dtype mapping, and a frozen parity-set checksum.
- `RuntimeVariantConverter`: one explicitly injected leaf with
  `convert(request) -> ConvertedPayload`. The leaf performs conversion and
  calculates the candidate's SHA-256 checksum.

### Process

The coordinator validates every identity and checksum before calling the leaf.
The source release must be `accepted`; its checkpoint checksum must equal its
immutable release checksum; converter status is exactly `official` or
`community`; and the parity-set checksum must be present. It calls the leaf
exactly once, with no retry. The returned payload must bind back to the same
source release and repeat the requested converter provenance and conversion
mapping. A successful `ConversionManifest` always says `parity_status:
pending`: conversion never makes a candidate selectable or approves parity.

### Outputs

- A `ConversionManifest` that binds the immutable source release to the
  checksummed candidate and the frozen parity set.

### Human check

Before runtime evaluation, compare the source-release and candidate checksums
in the manifest with the reviewed artifacts, then run the frozen parity set and
obtain a separate human approval. Do not register the runtime or publish a
bundle from this seam.

### Version and tests

MINOR (`v0.8.0`): additive Studio-only coordination capability; no existing
bundle or Reader seam changes and no migration is required.
`tests/voice_studio/test_runtime_conversion.py` uses only local in-memory fake
converters. It performs no conversion, filesystem payload work, subprocess,
model/runtime import, network, GPU, registry change, publication, retry, or
automatic parity approval.

## Bounded CUDA command provider

`cuda_command_provider.CudaCommandTrainingProvider` is the independently
deletable leaf that maps one provider-neutral `TrainingRequest` to one explicit
CUDA (graphics-processor) command plan. It implements the existing
`TrainingProvider.train(request, resume_from)` shape and returns one
`TrainedCheckpoint`.

### Inputs

- An approved `PreflightDecision` from the cost-control seam above.
- A frozen `TrainingCommandManifest`: explicit recipe id and revision, backend,
  argument-vector prefix, working directory, dataset/config/checkpoint paths,
  environment allowlist, target id, hard timeout, and captured-output cap.
- One injected `CommandRunner` implementing `run(CommandPlan) -> CommandResult`.

The accepted recipe identities are exact. Qwen3-TTS 12Hz 0.6B Base uses
`qwen3-tts-0.6b-base-official-cuda`; F5-TTS v1 uses
`f5-tts-v1-official-cuda`. Recipe, backend, and base-model identity must all
match. The adapter never substitutes Qwen for F5, F5 for Qwen, another model,
another device, or a local central-processor fallback.

### Process

Refuse a denied or contradictory preflight before the runner is called. Build
one argument tuple—never a shell command string—with explicit dataset,
configuration, base-model, output, and optional resume paths. A non-blocking
lock enforces one in-flight command per provider instance. The runner receives
the recipe provenance, concurrency limit `1`, timeout, and output cap.

Call the runner exactly once. There is no retry, fallback, polling,
provisioning, upload, provider application-programming interface (API), network,
subprocess, credential, conversion, promotion, Reader, or user-interface code
here. A local CUDA runner and a pre-provisioned rented-machine runner may both
implement the same injected protocol outside this module.

Timeout and cancellation are typed failures. Authentication, quota, rate-limit,
abuse, and repeated-error signals stop immediately as `CommandSafetyStop`.
Nonzero exit status, excessive captured output, unexpected runner exceptions,
and malformed results are also typed; none triggers another attempt.

### Outputs

- One immutable `CommandPlan` handed to the runner.
- One `TrainedCheckpoint` mapped from a valid successful `CommandResult`.

The runner owns actual execution and must enforce the supplied timeout,
cancellation, output capture, and environment. This module never claims those
controls ran merely because it placed them in the plan.

### Human check

Before allowing a real runner, print and inspect the `CommandPlan`. Confirm the
recipe revision, target, dataset/config paths, base model, environment entries,
timeout, cost approval, checkpoint directory, and resume path are the exact
approved values. Confirm no private path or credential appears unexpectedly.

### Version and tests

MINOR (`v0.7.0`): additive provider capability behind the existing training
interface; no schema migration. `tests/voice_studio/test_cuda_command_provider.py`
uses only injected in-memory runners and makes no command, network, vendor,
credential, installation, or GPU call.

## Optional standard-library process adapter

`stdlib_process_adapter` is an independently deletable host leaf. It supplies
the existing `ProcessFactory` and `ResultMapper` protocols to
`local_command_runner` without knowing a cloud provider, training backend, or
application. `StdlibProcessFactory` starts the already-approved argument vector
in a new process group and streams combined standard output and error as byte
chunks. Its `terminate` and `kill` operations signal that complete group, so a
cancelled training command does not leave child processes behind.

`LocalCommandRunner` remains the policy owner: it decides cancellation,
deadline, captured-output cap, and that there is only one attempt. The adapter
does not retry, provision a machine, upload data, call a network service, or
handle credentials. `JsonLineResultMapper` accepts exactly one UTF-8 JSON
checkpoint record, with no extra or repeated fields, and verifies the approved
checkpoint directory, lowercase SHA-256 checksum, and requested learned
artifact kind. The CUDA provider and remote-run coordinator validate the result
again at their own seams.

### Version and tests

MINOR (`v0.7.0`): optional host capability; no migration or artifact-schema
change. `tests/voice_studio/test_stdlib_process_adapter.py` injects fake process
creation, output pipes, and process-group signalling; it never starts a real
process or accesses a network, GPU, vendor service, or credential.
