---
type: module-contract
module: source-ingestion
sequence: semantic
---

# Source-ingestion contract

## Inputs

- Caller-provided `.txt` and `.docx` paths.
- A `.docx` is treated as an untrusted ZIP package and is checked against the
  limits declared in `parsers.py` before extraction.

## Process

- Validate the path and dispatch to the format-specific parser.
- Normalize extracted paragraphs and table cells to UTF-8 text.
- Split normalized text into sentence-sized recording units when requested by a
  caller; this module does not generate audio or invoke a model.

## Outputs

- Plain text or ordered sentence strings.
- Stable `FileNotFoundError`, `ValueError`, `ImportError`, or
  `NotImplementedError` at the adapter boundary.

## Human check

Open a normal DOCX and confirm paragraph/table order, then confirm malformed or
oversized packages are rejected without loading their contents into memory.
