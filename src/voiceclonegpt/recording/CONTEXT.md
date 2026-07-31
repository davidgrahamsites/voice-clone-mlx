# Recording module

One job: plan bounded, resumable processing windows for immutable recording
masters.

## Inputs

Validated recording duration and explicit chunk/overlap settings.

## Process

Create sequential windows with bounded overlap. Never read audio, call a model,
delete a source, or fan out work from this module.

## Outputs

Ordered `ChunkWindow` values with source-relative start and end times.

## Human check

Review a long-file plan and confirm every source second is covered while no
window exceeds the configured duration or repeats indefinitely.
