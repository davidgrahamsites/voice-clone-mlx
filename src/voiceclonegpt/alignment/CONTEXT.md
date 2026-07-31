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
