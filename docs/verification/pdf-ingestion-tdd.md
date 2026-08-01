# Verification — PDF ingestion (red/green record)

Adapter: `voiceclonemlx.ingestion.parsers._parse_pdf`.
Tests: `tests/voice_studio/test_pdf_ingestion.py`.

## Dependency situation

No PDF library is installed in this workspace, and this task permitted no
installs or network access:

```
$ python3 -c "import pypdf"     -> ModuleNotFoundError
$ python3 -c "import PyPDF2"    -> ModuleNotFoundError
$ python3 -c "import fitz"      -> ModuleNotFoundError
$ python3 -c "import pdfplumber"-> ModuleNotFoundError
```

So the adapter uses a **declared dependency seam**: `pypdf` is the chosen
library, imported lazily inside `_default_pdf_reader`, and `_parse_pdf` accepts
a `reader_factory` default argument. Tests drive the logic through fake readers
(`.pages`, each with `.extract_text()`), and two tests cover the real dependency
path by simulating an absent `pypdf`.

**This means the limits, rejections, and dispatch are verified; extraction
against a real PDF byte stream is not.** That gap is recorded in
`src/voiceclonemlx/ingestion/CONTEXT.md` under Human check.

## Red

```bash
$ python3 -m pytest tests/voice_studio/test_pdf_ingestion.py -q
18 failed, 1 passed
```

| Failing group | Why it failed |
|---|---|
| `TestPdfTextExtraction` (5) | `AttributeError: module 'parsers' has no attribute '_parse_pdf'` — no adapter existed; `.pdf` hit a `NotImplementedError("PDF support coming soon")` stub. |
| `TestPdfLimits` (4) | Same missing adapter, and no `MAX_PDF_FILE_BYTES` / `MAX_PDF_PAGES` / `MAX_PDF_TEXT_CHARS` constants to bound the work. `test_page_limit_checked_before_extraction` specifically pinned that the page cap must reject *before* extracting page text, not after. |
| `TestPdfRejections` (6) | No adapter, so no stable errors: malformed input, zero-page files, and scanned (text-free) PDFs had no defined behavior at all. |
| `TestPdfDependencySeam` (2) | No lazy `pypdf` import to fail cleanly; a missing library would have surfaced as a raw `ModuleNotFoundError` from wherever it was first touched. |
| `TestPdfDispatch::test_pdf_is_no_longer_not_implemented` | `AttributeError: no attribute '_default_pdf_reader'` — no injectable default reader. |

The one passing test was `test_missing_pdf_file_raises_file_not_found`: the
existing existence check already covered it, and it is kept as a regression
guard that the new dispatch did not move that check.

## Green

```bash
$ python3 -m pytest tests/voice_studio/test_pdf_ingestion.py -q
19 passed in 0.07s
```

Implementation added to `parsers.py`:

- Three `MAX_PDF_*` constants alongside the existing DOCX limits.
- `_default_pdf_reader(path)` — lazy `from pypdf import PdfReader`, raising a
  clear `ImportError` naming `pip install pypdf`.
- `_parse_pdf(path, reader_factory=None)` — file-size check, reader open with
  malformed input wrapped in a stable chained `ValueError`, no-pages check,
  page-count check **before** extraction, per-page text extraction (tolerating
  `None`), scanned-PDF rejection naming OCR and the .txt/.docx workaround, and
  a final extracted-text size check.
- `parse_source_file` dispatches `.pdf` to the adapter.

`ImportError` and `ValueError` are re-raised unwrapped inside the reader
`try` block so a missing dependency and a limit breach keep their own types
rather than being flattened into "not a readable PDF".

## Follow-up pass — bounded accumulation and error normalization

Two defects in the first implementation: page text was materialized for the
whole document before the size check ran (so an oversized PDF was fully
extracted before being rejected), and `ValueError` was re-raised unwrapped from
the reader block — meaning a `ValueError` thrown by the PDF *library* escaped
wearing the library's own message, indistinguishable from the adapter's
intentional limit errors.

### Red

```bash
$ python3 -m pytest tests/voice_studio/test_pdf_ingestion.py -q
3 failed, 21 passed
```

| Failing test | Why it failed |
|---|---|
| `TestPdfLimits::test_stops_extracting_once_text_limit_exceeded` | All 50 pages were extracted before the limit was applied; the test asserts only pages `[0, 1]` are touched before rejection. |
| `TestPdfRejections::test_reader_value_error_is_normalized` | A library `ValueError` propagated verbatim (`invalid literal for int()...`) instead of the stable `File is not a readable PDF`. |
| `TestPdfRejections::test_reader_value_error_preserves_cause` | Same: `__cause__` was `None` because nothing wrapped it. |

`test_page_extraction_value_error_is_normalized` and
`test_adapter_limit_errors_are_not_normalized` passed on first run — kept as
regression guards for the two behaviors that were already correct.

### Green

```bash
$ python3 -m pytest tests/voice_studio/test_pdf_ingestion.py -q
24 passed in 0.08s
```

Changes to `_parse_pdf`:

- Pages are extracted in a loop, appended to a `chunks` list with a running
  `total_chars`; the limit is checked after each page and raises immediately.
  The redundant post-join size check was removed so the rule has one home.
- The reader block now re-raises only `ImportError`; every other exception —
  `ValueError` included — becomes a chained `File is not a readable PDF`.
- The per-page limit check sits **outside** the per-page extraction `try`, so
  the adapter's own limit error can never be caught and renormalized into
  "not a readable PDF". A test asserts it carries no `__cause__`.

## Test-contract update — obsolete stub test removed

Shipping the adapter left one test asserting the old contract:

```bash
$ python3 -m pytest -q
1 failed, 135 passed
FAILED tests/voice_studio/test_ingestion.py::TestTextParsing::test_pdf_format_not_yet_implemented
```

`test_pdf_format_not_yet_implemented` asserted that `.pdf` raises
`NotImplementedError`. That is no longer the contract: `.pdf` dispatches to
`_parse_pdf`, and its real behavior is covered by
`tests/voice_studio/test_pdf_ingestion.py` (24 tests, including dispatch and
the missing-dependency path). The test was deleted — the same treatment
`test_docx_format_not_yet_implemented` received when DOCX shipped.

This is a **deliberate contract change, not a failing test made to pass**:
the assertion described a stub that no longer exists. No other test in
`test_ingestion.py` was touched (31 tests remain), and no production code
changed.

```bash
$ python3 -m pytest tests/voice_studio/test_ingestion.py tests/voice_studio/test_pdf_ingestion.py -q
55 passed in 0.55s
$ python3 -m pytest -q
140 passed in 4.51s
```
