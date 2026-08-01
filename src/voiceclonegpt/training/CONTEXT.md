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
