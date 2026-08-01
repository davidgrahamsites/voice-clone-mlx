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
| `http://`, `https://` | `web.fetch_url` → `web_html.extract_text` | stdlib `urllib` + `html.parser` |
| anything else | rejected | — |

URLs are detected before any filesystem lookup (`web.is_url`), so a URL is
never reported as a missing file.

## Process

`parse_source_file(path)` checks existence, dispatches on suffix, and returns
text. Each format is its own small adapter; the dispatcher holds no format
logic.

**Web ingestion is four modules, split by responsibility:**

| Module | Job | Knows about |
|---|---|---|
| `web.py` | Public `fetch_url`; URL and destination policy, content type, limits, error contract | the three modules below |
| `web_transport.py` | urllib wiring: redirect refusal, pinned connections, opener assembly | sockets, HTTP |
| `web_body.py` | Bounded, deadline-aware body reads | anything with `.read(size)` |
| `web_html.py` | HTML in, readable text out | nothing but HTML |

Dependencies run **one way only**: `web` → `web_transport`, `web_body`,
`web_html`. None of the three imports `web`, so each can be tested — and
replaced — on its own. `web.py` decides policy; the others carry it out and
hold no policy of their own. `web_transport` dials the address `web` approved;
it does not validate.

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
| `web.MAX_WEB_RESPONSE_BYTES` | 5 MB | oversized / endless responses |
| `web.MAX_WEB_TEXT_CHARS` | 5M | runaway extraction |
| `web.WEB_TIMEOUT_SECONDS` | 10 | hung connections and slow-drip reads |
| `web_body.READ_CHUNK_BYTES` | 64 KB | unbounded single reads |

DOCX limits are checked from ZIP **metadata before decompressing anything**.
PDF file size is checked before opening; page count before extracting any page.
Web responses are read with an explicit byte cap (`read(limit + 1)`), never
unbounded, and URL validation runs before any connection is opened.

**Web scope is deliberately narrow.** `web.fetch_url` makes exactly one
request for exactly the URL given. It does not follow links, retry, crawl, or
issue a second request under any condition — the guard against burst traffic is
structural, not a rate limit. **Redirects are rejected, not followed**: the
urllib opener is built without redirect handling, and any 3xx response is
refused before its body is read. Supply the final URL yourself.

**Destination allowlisting.** Only `http`/`https` are fetchable; `file:`,
`ftp:`, `data:`, and `javascript:` are rejected before the opener is called.
Before connecting, the destination is checked: an IP literal directly, a
hostname by resolving it and requiring **every** returned address to pass.
Loopback, private, link-local (including cloud metadata at `169.254.169.254`),
multicast, unspecified, and reserved ranges are refused, as are their
IPv4-mapped IPv6 forms. This makes the adapter unusable as an SSRF pivot into
localhost or a private network. Only textual content types are accepted.

**The validated address is the one dialed.** The host is resolved exactly once;
the approved address is pinned and the connection dials it directly, so a
second DNS answer cannot redirect the connection after validation (DNS
rebinding). Pinning changes only *where* we connect — the `Host` header and TLS
SNI keep the real hostname, so certificate verification is unaffected.

**Time bound.** The body is read in 64 KB chunks against a wall-clock deadline
derived from `timeout`. Before each chunk the remaining deadline is applied to
the underlying socket, so a single blocking read cannot outlast it either — not
just the gaps between reads. Clock, resolver, transport, and the timeout setter
are all injectable.

**Seams.** `_parse_pdf(path, reader_factory=None)` takes its reader as a
default argument, so every limit and rejection is testable without `pypdf`
installed. `_default_pdf_reader` is the only place that imports it.
`web.fetch_url(url, opener=None, timeout=None, resolver=None, clock=None,
read_timeout_setter=None)` does the same for the network — transport, DNS, and time are all injectable, so
**no test in this project makes a live request or a real DNS lookup** (an
autouse fixture in the web tests patches the resolver, so a test that tried
would fail rather than silently reach the network). `web.extract_text(html)` is
a pure function, testable with a string.

## Outputs

A UTF-8 string. DOCX joins paragraph text then table cell text, one per line,
blanks dropped. PDF joins non-empty page text, one page per line, blanks
dropped. Web pages yield readable text only — `script`, `style`, `noscript`,
`template`, `svg`, and `head` content is discarded, markup and link hrefs are
dropped while link *text* is kept, HTML entities are decoded, block boundaries
become line breaks, and whitespace within a line collapses.
`split_into_sentences(text)` then splits on `.`/`!`/`?` followed by
whitespace — a deliberately simple splitter that does not special-case
abbreviations.

**Errors are stable and typed** — callers never see library internals:

- `FileNotFoundError` — file does not exist
- `ValueError` — unsupported suffix; any limit exceeded; malformed or
  unreadable package; PDF with no pages; PDF with no extractable text
  (scanned — the message names OCR and the .txt/.docx workaround);
  non-http(s) URL scheme; URL with no host; blocked destination address;
  unresolvable host; redirect response (message names the status and says to
  supply the final URL); HTTP status failure (message carries the code);
  unreachable host, connect timeout, or read deadline exceeded; unsupported
  content type; page with no readable text
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
5. **Web extraction has never run against a live site**, by design — no test
   makes a network request. Before ingesting from a URL for real, fetch one
   page manually and read the output: boilerplate (navigation, cookie banners,
   footers) is *not* stripped, so it will end up in the recording script unless
   you edit it out.
6. **Check you have the right to use the page's text** before recording it.
   The adapter does not read `robots.txt` or check terms of use.
7. **If a URL is rejected as a redirect**, fetch the final URL and pass that.
   Do not add redirect following to work around it — one call is one request
   by design.
8. **Do not relax the destination block list to reach an internal service.**
   Copy the content to a local file and ingest that instead; the block list is
   what stops this adapter being used to reach localhost or cloud metadata.

## Tests

- `tests/voice_studio/test_ingestion.py` — TXT, DOCX, DOCX preflight, splitting
- `tests/voice_studio/test_pdf_ingestion.py` — PDF extraction, limits,
  rejections, dependency seam, dispatch
- `tests/voice_studio/test_web_html.py` — HTML extraction (pure)
- `tests/voice_studio/test_web_transport.py` — URL validation, response limits,
  content types, transport errors, redirects, single-request guarantee,
  dispatch
- `tests/voice_studio/test_web_security.py` — destination allowlisting (SSRF),
  read deadline, per-read socket timeout
- `tests/voice_studio/test_web_pinning.py` — address pinning: the validated
  address is the one dialed, with Host and SNI intact
- `tests/voice_studio/web_test_support.py` — shared fakes and the autouse
  `no_live_dns` fixture. Import it into any new web test module: it is what
  guarantees a test cannot reach DNS or the network.
- Red/green evidence: `docs/verification/pdf-ingestion-tdd.md`,
  `docs/verification/web-ingestion-tdd.md`
