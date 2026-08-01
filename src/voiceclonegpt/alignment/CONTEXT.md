# Alignment module

One job: turn provider-neutral diarization spans into a safe dataset admission
decision for one candidate clip.

## Inputs

- Candidate clip start/end offsets.
- Enrolled target speaker id.
- Diarization `SpeakerTurn` spans and overlap flags.

## Process

Ignore turns outside the candidate range. Reject the entire candidate when an
overlap flag is present, when another speaker intersects it, or when the target
speaker is absent. Accept only target-only, non-overlapping clips.

## Outputs

A `ClipDecision` with `accept` or `reject` plus a stable reason. This module
does not load models, edit audio, transcribe, or write manifests.

## Human check

Listen to accepted and rejected fixture clips and confirm that every mixed-
speaker clip is rejected before dataset construction.

## Style markers

`marker_parser.py` owns the **wording** of a spoken style marker — the one home
for it. It has two directions:

- `parse_style_marker(text) -> StyleMarker | None` — an utterance to a
  canonical style.
- `marker_phrase(style) -> str` — a canonical style to the exact sentence the
  reader is asked to say, and the teleprompter prints.

Both are pure: text in, text out. No audio, model, filesystem, network, or UI.
Standard library only.

### A marker is a whole utterance

`parse_style_marker` now anchors with `fullmatch`. It was `search`, which
matched a marker **buried inside ordinary speech** — "okay, this is the neutral
reading, let us begin" was accepted as a style boundary. That silently
relabels real speech as a marker, and the mislabelled audio then enters the
dataset. Tolerated: surrounding whitespace, case, an optional `the`/`a`,
trailing `.`/`!`/`?`, and repeated inner spaces. Other words are not.

### Only `netural` is aliased

That is the misspelling the reader actually produces. `nuetral`, `neutrl`,
`netrual` and friends return None rather than being guessed at — a fuzzy match
mislabels an entire recorded block, and a wrong style is worse than an
unrecognized one, because a human reviews the second and not the first.

### Behavior change

Rejecting embedded prose is a **narrowing** of what `parse_style_marker`
accepts. Nothing on `main` calls it outside its tests today (verified with
`git grep`), so no caller breaks — but if a future caller relies on finding a
marker inside a longer transcript segment, it must split into utterances first
rather than have this function loosened.

### TDD evidence

```bash
# red — the generator did not exist and embedded prose was accepted
$ python3 -m pytest tests/voice_studio/test_marker_parser.py -q
ERROR — ImportError: cannot import name 'marker_phrase'

# green — the original 4 tests are unchanged and still pass
$ python3 -m pytest tests/voice_studio/test_marker_parser.py -q
64 passed in 0.06s
$ python3 -m pytest -q
694 passed, 10 skipped in 5.94s
$ git diff --check
(clean)
```

`tests/voice_studio/test_marker_parser.py` (64) — phrase generation for all
eight styles, round-trip through the generated phrase, the `netural` alias in
both directions, case/article/punctuation/whitespace variants, embedded-prose
rejection, unknown styles, non-string input, and the frozen result.
