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
