---
type: test-contract
module: voice-reader-tests
---

# Voice Reader tests

## Inputs

- Synthetic immutable voice-model bundles created inside each test temporary
  directory.

## Process

- Exercise provider-neutral bundle parsing, checksum verification, and explicit
  multi-voice selection without loading backend weights.

## Outputs

- Repeatable pytest results and fixtures that remain independent of Voice Studio.

## Human check

Confirm that tests cover both the user's voice and a second selectable voice,
and that a tampered bundle fails before model initialization.
