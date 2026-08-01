# Voice Studio seam tests

These tests cover the small Voice Studio seams currently implemented here:
source ingestion and sentence splitting, script generation, cost preflight,
marker parsing, chunk planning, and the provider-neutral overlap gate.

## Inputs

Temporary `.txt`/`.docx` fixtures, synthetic `SpeakerTurn` spans, and bounded
cost/script inputs. Tests never call a remote service or mutate source data.

## Process

Run the tests with `PYTHONPATH=src` and assert each seam's stable contract,
including the hard rule that any mixed-speaker candidate clip is rejected.

## Outputs

Regression evidence for all listed seams, including malformed/oversized DOCX
rejection and overlap rejection.

## Human check

Review that fixtures represent the intended hard rules before adding a real
diarizer or model-backed adapter.
