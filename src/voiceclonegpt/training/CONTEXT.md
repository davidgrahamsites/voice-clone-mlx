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
