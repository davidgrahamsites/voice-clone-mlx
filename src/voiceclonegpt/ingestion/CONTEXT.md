# Ingestion — source files to normalized text

Turns untrusted source documents into a plain UTF-8 string, and splits that
string into sentences. Everything downstream (script generation, recording)
treats this folder's output as the only view of the source material.

## Inputs

Paths to files supplied by the user. **All input is untrusted** — a source file
may be hostile (zip bomb, malformed package) or simply unusable (scanned PDF).

| Suffix | Adapter | Dependency |
|---|---|---|
| `.txt` | `parse_source_file` (direct read) | none |
| `.docx` | `_preflight_docx` + `_parse_docx` | `python-docx` (installed) |
| `.pdf` | `_parse_pdf` via `_default_pdf_reader` | `pypdf` (**not installed here**) |
| anything else | rejected | — |

## Process

`parse_source_file(path)` checks existence, dispatches on suffix, and returns
text. Each format is its own small adapter; the dispatcher holds no format
logic.

**Limits** (module constants, one home each, monkeypatchable in tests):

| Constant | Default | Guards against |
|---|---|---|
| `MAX_DOCX_FILE_BYTES` | 50 MB | oversized upload |
| `MAX_DOCX_ENTRIES` | 2000 | entry-flood archives |
| `MAX_DOCX_UNCOMPRESSED_BYTES` | 200 MB | decompression blowup |
| `MAX_DOCX_COMPRESSION_RATIO` | 200× | zip bombs |
| `MAX_DOCX_TEXT_CHARS` | 5M | runaway extraction |
| `MAX_PDF_FILE_BYTES` | 50 MB | oversized upload |
| `MAX_PDF_PAGES` | 2000 | page-flood documents |
| `MAX_PDF_TEXT_CHARS` | 5M | runaway extraction |

DOCX limits are checked from ZIP **metadata before decompressing anything**.
PDF file size is checked before opening; page count before extracting any page.

**Seams.** `_parse_pdf(path, reader_factory=None)` takes its reader as a
default argument, so every limit and rejection is testable without `pypdf`
installed. `_default_pdf_reader` is the only place that imports it.

## Outputs

A UTF-8 string. DOCX joins paragraph text then table cell text, one per line,
blanks dropped. PDF joins non-empty page text, one page per line, blanks
dropped. `split_into_sentences(text)` then splits on `.`/`!`/`?` followed by
whitespace — a deliberately simple splitter that does not special-case
abbreviations.

**Errors are stable and typed** — callers never see library internals:

- `FileNotFoundError` — file does not exist
- `ValueError` — unsupported suffix; any limit exceeded; malformed or
  unreadable package; PDF with no pages; PDF with no extractable text
  (scanned — the message names OCR and the .txt/.docx workaround)
- `ImportError` — optional dependency missing, naming the pip command

Every wrapped failure chains the original exception on `__cause__`.

## Human check

1. **After ingesting a new document**, read the first and last few sentences of
   the output. Table-heavy DOCX and multi-column PDF extract in reading order
   that may not match the visual layout.
2. **If a PDF is rejected as scanned**, that is a real finding, not a bug —
   re-export it as text or convert to .docx rather than raising the limits.
3. **Before raising any limit constant**, confirm the file is trusted. The
   limits exist to bound hostile input, not to be tuned per document.
4. **PDF requires `pypdf`**, which is not installed in this workspace. The
   adapter is exercised through its reader seam; before relying on PDF in
   production, install `pypdf` and parse one real document end to end.

## Tests

- `tests/voice_studio/test_ingestion.py` — TXT, DOCX, DOCX preflight, splitting
- `tests/voice_studio/test_pdf_ingestion.py` — PDF extraction, limits,
  rejections, dependency seam, dispatch
- Red/green evidence: `docs/verification/pdf-ingestion-tdd.md`
