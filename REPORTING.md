# Plain-language reporting policy

All project reporting must be understandable without specialist knowledge.

## Required style

- Every status update begins with the current estimated percentage of the
  entire project that is finished. Keep the estimate honest and change it only
  when verified work changes the remaining path to the real voice-model round
  trip.
- Lead with what happened, what it means, and what happens next.
- Use ordinary words whenever they are accurate.
- Define every unavoidable technical term inline the first time it appears.
  Example: “speaker diarization (marking who spoke when).”
- Expand an abbreviation on first use, then use the short form only if it
  improves readability.
- Report failures with the concrete cause, affected scope, user impact, and
  recovery action.
- Include exact paths, versions, commands, and test counts when they help
  someone reproduce the result, but do not hide the conclusion in a log dump.
- Do not use unexplained model names, acronyms, internal worker names, or
  jargon as if they were self-explanatory.

## Worker handoff

Every handoff states, in plain talk: what changed, what did not change, the
version/branch impact, verification performed, unresolved risk, and the next
decision needed. If a technical term is necessary, define it in the same
sentence where it first appears.
