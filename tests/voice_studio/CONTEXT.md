# Voice Studio alignment tests

One job: verify the provider-neutral overlap gate's acceptance and rejection
contract.

## Inputs

Synthetic `SpeakerTurn` fixtures covering target-only, overlap, other-speaker,
and out-of-range spans.

## Process

Run the tests with `PYTHONPATH=src` and assert stable `ClipDecision` values.

## Outputs

Regression evidence that any mixed-speaker candidate clip is rejected.

## Human check

Review that the fixtures represent the intended hard rule before adding a real
diarizer adapter.
