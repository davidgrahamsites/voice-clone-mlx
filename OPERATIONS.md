# External-service safety policy

This workspace is local-first and deliberately conservative with APIs,
external services, websites, and hosted model/GPU providers.

## Non-negotiable rule

**Never burst APIs, services, or web apps.** This applies to every worker,
script, app, browser automation flow, research job, model download, webhook,
local data-bus client, and retry loop.

Before making external calls, a worker or module must:

1. identify the service's published rate/concurrency limits;
2. set an explicit low concurrency cap (default: one request at a time);
3. enforce a request interval, timeout, response-size limit, and total work
   budget;
4. cache or checkpoint reusable results so reruns do not repeat calls;
5. use exponential backoff with jitter for transient failures; and
6. stop immediately on `429`, quota, authentication, abuse, or repeated-error
   responses and surface a typed, actionable error.

Do not use unbounded `gather`/fan-out, tight polling, automatic infinite
retries, page-refresh loops, or simultaneous model downloads. A user-approved
batch still needs a queue, a progress record, a cancel path, and bounded
parallelism. If a service does not publish limits, use one-at-a-time calls and
an intentionally generous delay until measured behavior supports a safer
limit.

## Local-first exceptions

Local CPU/MLX work may be parallelized only after measuring memory and thermal
headroom. Loopback services still need bounded queues and request sizes. Never
send private recordings, documents, model weights, or generated text to an
external service unless the user explicitly chooses that provider for that
operation.

## Worker handoff

Every task that touches an external service reports the service, concurrency
cap, delay/backoff policy, cache/checkpoint location, total budget, and the
verification used to prove that the safety rule is active.

Workers are read-only until the orchestrator approves a specific write scope.
This includes filesystem edits, branch changes, commits, dependency installs,
uploads, POST/PUT/PATCH/DELETE requests, hosted-job creation, and changes to a
local or remote service. A write request must include exact scope, purpose,
version/branch impact, privacy/data impact, verification, and rollback. The
orchestrator must answer with the literal phrase `APPROVED WRITE` plus the
scope. A worker must stop if the requested operation exceeds that scope.

## Git safety

**Never burst Git.** Run Git commands one at a time, with a deliberate pause
between repeated operations when a remote or large repository is involved.
Never fan out `fetch`, `pull`, `push`, clone, checkout, status, log, or branch
commands across workers or terminals. Do not run tight Git polling loops or
automatic retry storms. Cache inspection results, use bounded commands, and
stop on authentication, lock, rate-limit, or repository-corruption errors.

Before any remote Git write (`push`, tag publication, release, or pull-request
operation), obtain a separate scoped `APPROVED WRITE` that names the exact
repository, branch/tag, and rollback. Local commits and branch changes still
follow the read-only worker rule and version policy.
