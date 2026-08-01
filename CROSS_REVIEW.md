# Optional independent review checklist

Independent review is optional. Use it when the user explicitly requests it or
when the orchestrator identifies a high-risk change involving security,
destructive operations, external services, model formats, artifact migrations,
or another difficult-to-reverse boundary. Ordinary changes advance through
focused tests and the milestone ICM gate without a cross-family review.

## Reviewer routing

- Code written by Claude or a Claude worker is reviewed by Codex `gpt-5.6-sol`
  (“Sol”).
- Code written by Codex or a Codex worker is reviewed by Claude Haiku 4.5.
- The reviewer is read-only. It reports findings and does not silently edit the
  author's files.
- If an optional review is requested and the assigned reviewer is unavailable,
  the orchestrator decides whether to wait or record why the review was waived.
- If a finding needs a fix, the fixer submits a scoped write request and waits
  for `APPROVED WRITE`. The opposite agent family reviews the resulting diff
  again.

## Review questions when this checklist is used

The reviewer answers these questions in plain language:

1. **Security:** Can untrusted text, audio, paths, URLs, model files, or
   service responses cause data leaks, command execution, unsafe file access,
   secret exposure, denial of service, or unauthorized network activity?
2. **Purpose:** Does the change do the named job, preserve the contract, and
   avoid unrelated behavior or hidden scope growth?
3. **Five whys:** For every defect or risk, ask “why?” five times—or until the
   underlying process/design cause is reached—and record the cause and fix.
4. **God object:** Does one module own too many responsibilities, such as UI,
   workflow decisions, model calls, filesystem writes, retries, and policy?
   If yes, identify the clean seams that should be split.
5. **Test-driven development:** Is there a test for the intended behavior that
   failed before the implementation, then passed after the smallest change?
   Missing tests or skipped red/green evidence block approval unless the change
   is documentation-only.

## Review record

The review record names the author family, reviewer model, branch, version
impact, files inspected, tests run, `/icm-check` result, findings by severity,
five-whys analysis for each finding, and the final decision (`approved`,
`changes_requested`, or `blocked`). Define technical terms the first time they
appear. Store the record beside the change or in the task handoff; do not place
private audio, transcripts, secrets, or model weights in it.
