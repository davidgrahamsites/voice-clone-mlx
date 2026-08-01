# Verification — website ingestion (red/green record)

Adapter: `src/voiceclonemlx/ingestion/web.py`.
Tests: `tests/voice_studio/test_web_html.py` (extraction) and
`tests/voice_studio/test_web_transport.py` (everything else).

**Note on history below:** passes 1–3 ran against a single
`tests/voice_studio/test_web_ingestion.py`, split into the two files above in
the final pass. Commands quoted in those sections are the ones actually run at
the time and are left unedited as a record; use the two current paths today.

## No live network, by construction

`fetch_url(url, opener=None, timeout=None)` takes its opener as a default
argument. Every test injects a fake response object; the two dispatch tests
monkeypatch `web._default_opener`. **No test in this file opens a socket.**
Only stdlib is used — `urllib.request` for transport, `html.parser` for
extraction — so there is no new dependency to declare.

## Red

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
ERROR tests/voice_studio/test_web_ingestion.py
ImportError: cannot import name 'web' from 'voiceclonemlx.ingestion'
1 error in 0.18s
```

Collection failed outright: no adapter existed. The 34 tests written first
specify the whole contract before any of it was implemented:

| Group | What it pins down |
|---|---|
| `TestUrlValidation` (5) | `file:`, `ftp:`, `javascript:`, `data:` rejected; scheme check is case-insensitive; a URL with no host rejected; and validation happens **before** the opener is called (asserted by a recording opener that must see zero calls). |
| `TestHtmlExtraction` (8) | Visible text kept; `script`/`style`/`noscript` content dropped; no markup characters survive; adjacent blocks do not run together; entities decoded; whitespace collapsed; `extract_text` usable as a pure function; text-free page rejected. |
| `TestResponseLimits` (5) | Oversized response rejected; `read()` is always called **with a size argument** (never unbounded); oversized extracted text rejected; the configured timeout is passed to the opener and is overridable. |
| `TestContentType` (3) | `text/plain` accepted; `application/pdf` rejected; a missing Content-Type is tolerated as HTML. |
| `TestTransportErrors` (5) | `HTTPError` becomes a `ValueError` naming the status code; `URLError` and `TimeoutError` become `Could not fetch`; the cause is chained; the adapter's own errors are *not* normalized and carry no `__cause__`. |
| `TestNoRecursion` (2) | Exactly one opener call for one URL; link text kept, `href` target discarded. |
| `TestDispatch` (3) | `parse_source_file` routes URLs to the adapter, does not treat a URL as a missing file, and still handles local paths. |

## Green

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
34 passed in 0.11s
$ python3 -m pytest -q
174 passed in 5.56s
```

### One genuine bug caught on the first run

The first implementation passed 33 of 34. `test_collapses_whitespace` failed:

```
assert 'Lots of space' in 'Lots\nof space'
```

The extractor used `"\n"` to mark block boundaries and then split the document
on `"\n"` — so newlines that were merely *source formatting inside* a paragraph
were treated as block breaks, chopping one sentence into several lines. Since
`split_into_sentences` runs downstream on this text, that would have fractured
sentences mid-phrase in the recording script.

Fix: block boundaries are marked with a `BLOCK_SEPARATOR` sentinel (`"\x00"`)
that cannot occur in readable page text; the split happens on that sentinel,
and all real whitespace — newlines included — collapses within a line. The test
expectation was correct and the code was wrong; the implementation changed, not
the test.

## Security hardening pass

The first adapter had three holes: urllib followed redirects (so "one request"
was only true when the server cooperated), any resolvable host could be
fetched (including `127.0.0.1` and cloud metadata at `169.254.169.254`), and
the body read was bounded in bytes but not in time (a server dribbling bytes
below the cap could hold the read open indefinitely).

### Red

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
69 errors in 0.88s
AttributeError: module 'voiceclonemlx.ingestion.web' has no attribute '_default_resolver'
```

Every test errored, including the pre-existing ones: the new autouse
`no_live_dns` fixture patches `web._default_resolver`, which did not exist.
That fixture is the structural guarantee that **no test in this module can
resolve a hostname for real** — it fails loudly rather than silently reaching
DNS.

| New group | What it pins down |
|---|---|
| `TestRedirects` (8) | Every 3xx status (301/302/303/307/308) rejected; an `HTTPError` with a 3xx code reported as a redirect rather than a generic HTTP failure; the redirect body is **never read**; `_NoRedirect.redirect_request` returns `None`; the built opener carries that handler. |
| `TestDestinationValidation` (20) | 13 literal addresses rejected — loopback (`127.0.0.1`, `127.9.9.9`), private (`10.x`, `192.168.x`, `172.16.x`), link-local/metadata (`169.254.169.254`), unspecified (`0.0.0.0`), multicast (`224.0.0.1`), and the IPv6 forms `::1`, `fe80::1`, `fc00::1`, `::`, plus the IPv4-mapped `::ffff:127.0.0.1`; a public literal still allowed; hostnames rejected by *resolved* address; a host resolving to both public and private addresses rejected; unresolvable and empty resolutions rejected; validation runs before the opener; the resolver is injectable as a parameter. |
| `TestReadDeadline` (5) | A response that never reaches EOF is abandoned once the deadline passes; a read inside the deadline succeeds; the deadline is derived from the caller's `timeout`; the byte cap still applies independently; a transport `ValueError` during read is normalized, not mistaken for the deadline. |

### Green

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
70 passed in 0.19s
$ python3 -m pytest -q
210 passed in 5.49s
```

Implementation:

- `_NoRedirect` / `_build_opener` — `_default_opener` now uses a private opener
  with redirect following removed, so urllib raises `HTTPError` on 3xx instead
  of chasing it. `fetch_url` additionally checks `response.status` and rejects
  any 3xx **before reading the body**, which covers injected openers too.
- `_default_resolver`, `_is_blocked_address`, `_validate_destination` — IP
  literals are checked directly; hostnames are resolved and *every* returned
  address must pass. `_is_blocked_address` unwraps `ipv4_mapped` first, so
  `::ffff:127.0.0.1` cannot smuggle a loopback address past the IPv6 checks.
- `_read_bounded(response, limit, deadline, clock)` — reads in 64 KB chunks,
  checking the clock before each, and stops at `limit + 1` bytes. `clock` is
  injectable, so the slow-response tests are deterministic and instant.

### A defect found in my own first cut of this pass

The read block was written as `except ValueError: raise` to let the deadline
error through — which meant a `ValueError` raised by the *transport* would
also escape unnormalized, exactly the defect flagged earlier on the PDF
adapter. Replaced with a private `_ReadDeadlineExceeded` exception that
`fetch_url` converts, so transport `ValueError`s still normalize to
`Could not fetch`. `test_transport_value_error_during_read_is_normalized`
locks it.

## Second hardening pass — per-read timeout and DNS pinning

Two holes survived the first pass:

- The deadline was only checked **between** chunks. A server that accepted the
  connection and then never sent a byte would block inside a single
  `response.read()` for as long as the socket allowed, and the deadline check
  would not run again until that read returned.
- Validation resolved the hostname, then urllib resolved it **again** when
  connecting. Between those two lookups the answer can change (DNS rebinding),
  so a host could validate as public and connect to `127.0.0.1`.

### Red

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
12 failed, 70 passed in 0.43s
```

| Failing test | Why it failed |
|---|---|
| `TestPerReadTimeout::test_socket_timeout_is_set_before_each_read` | `fetch_url` had no `read_timeout_setter` parameter; nothing was ever applied to the socket. |
| `…::test_applied_timeout_shrinks_as_the_deadline_approaches` | Same — no per-read bound existed, so there was no shrinking sequence to observe. |
| `…::test_single_blocking_read_cannot_outlast_the_deadline` | A read that blocks past the deadline was not cut off; only the between-chunk check existed. |
| `…::test_default_setter_tolerates_a_response_without_a_socket` | `_apply_read_timeout` did not exist. |
| `…::test_default_setter_sets_the_underlying_socket_timeout` | Same — nothing reached `fp.raw._sock`. |
| `TestAddressPinning::test_validated_address_is_pinned_for_the_connection` | `_default_opener` took no `pinned_ip`; urllib resolved the host itself. |
| `…::test_second_resolution_cannot_bypass_the_block` | **The TOCTOU proof.** The rebinding resolver returns a public address first and `127.0.0.1` after; the old code validated the first answer and let urllib resolve again at connect time. |
| `…::test_ip_literal_is_pinned_to_itself` | No pinning path at all. |
| `…::test_validate_destination_returns_the_address_it_approved` | `_validate_destination` returned `None`, so the approved address was discarded. |
| `…::test_pinned_connection_keeps_the_hostname_for_host_and_sni` | `_pinned_connection_factory` did not exist. |
| `…::test_pinned_https_connection_keeps_the_hostname` | Same, for the TLS path. |
| `…::test_pinned_opener_is_still_redirect_free` | `_build_opener` took no `pinned_ip`. |

### Green

```bash
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
82 passed in 0.24s
$ python3 -m pytest -q
222 passed in 5.38s
```

Implementation:

- `_apply_read_timeout(response, seconds)` walks `fp.raw._sock` → `fp._sock` →
  `_sock` and calls `settimeout` on the first socket it finds, doing nothing if
  the response exposes none (test fakes, exotic transports). `_read_bounded`
  computes `remaining = deadline - clock()` and applies it before every chunk,
  so the socket itself enforces the remainder of the overall deadline. The
  setter is injectable via `fetch_url(read_timeout_setter=...)`.
- `_validate_destination` now **returns** the address it approved, and
  `fetch_url` passes it to `_default_opener(url, timeout, pinned_ip=...)`.
  `_pinned_connection_factory` builds an `HTTP(S)Connection` subclass whose
  `connect()` dials the pinned address while leaving `self.host` as the
  hostname — so the `Host` header and TLS `server_hostname` (SNI, and therefore
  certificate verification) are unchanged. `_PinnedHTTPHandler` /
  `_PinnedHTTPSHandler` install it, and `_build_opener(pinned_ip)` keeps
  `_NoRedirect` in the chain.

The host is now resolved exactly **once** per fetch;
`test_second_resolution_cannot_bypass_the_block` asserts both that the resolver
is called a single time and that the connection used the validated address.

### Two existing tests updated

`TestDispatch`'s two monkeypatched openers replaced `_default_opener`, whose
signature gained `pinned_ip`. Their lambdas now accept `pinned_ip=None`. No
assertion changed — this is a signature update, not a weakened expectation.

## Structural split — pure extraction separated from transport

No behavior change. `web.py` had grown to 540 lines mixing HTML parsing with
socket policy, and its test file to 800 lines covering both.

### Baseline (before)

```bash
$ python3 -m pytest -q
222 passed in 5.52s
$ python3 -m pytest tests/voice_studio/test_web_ingestion.py -q
82 passed in 0.20s
$ wc -l src/voiceclonemlx/ingestion/web.py tests/voice_studio/test_web_ingestion.py
540  src/voiceclonemlx/ingestion/web.py
800  tests/voice_studio/test_web_ingestion.py
```

The 82 baseline test IDs were captured to disk before any edit, and the
post-split IDs diffed against them.

### After

```bash
$ python3 -m pytest tests/voice_studio/test_web_html.py tests/voice_studio/test_web_transport.py -q
87 passed in 0.29s
$ python3 -m pytest -q
227 passed in 5.47s
```

### Source mapping

| Moved out of `web.py` | Into `web_html.py` |
|---|---|
| `SKIPPED_ELEMENTS`, `BLOCK_ELEMENTS`, `BLOCK_SEPARATOR` | verbatim |
| `_TextExtractor` | verbatim |
| `extract_text` | verbatim |
| `import re`, `html.parser`, `html.unescape` | moved with them |

`web.py` keeps `is_url`, `_validate_url`, `_default_resolver`,
`_is_blocked_address`, `_validate_destination`, `_NoRedirect`,
`_pinned_connection_factory`, `_PinnedHTTPHandler`, `_PinnedHTTPSHandler`,
`_build_opener`, `_default_opener`, `_apply_read_timeout`,
`_ReadDeadlineExceeded`, `_read_bounded`, `_content_type`, `fetch_url`, and
every limit constant. It imports `extract_text` from the pure module — a
one-way dependency.

### Test mapping

| Baseline class (82 tests) | Destination |
|---|---|
| `TestHtmlExtraction` (7 pure tests) | `test_web_html.py`, same names, now calling `extract_text` directly instead of through `fetch_url` |
| `TestHtmlExtraction::test_rejects_page_with_no_readable_text` | `test_web_transport.py::TestExtractionIsWiredIn` — it asserts *fetch policy* (empty page is an error), not extraction |
| `TestUrlValidation`, `TestResponseLimits`, `TestContentType`, `TestTransportErrors`, `TestNoRecursion`, `TestRedirects`, `TestDestinationValidation`, `TestReadDeadline`, `TestPerReadTimeout`, `TestAddressPinning`, `TestDispatch` | `test_web_transport.py`, verbatim — class and test names unchanged |
| shared fixtures (`FakeResponse`, `opener_of`, `PAGE`, autouse `no_live_dns`) | `test_web_transport.py`; `test_web_html.py` needs none of them |

Test-ID diff against the captured baseline: **nothing lost**. One rename
(`test_rejects_page_with_no_readable_text` moved class), and six additions:

- `TestExtractionIsWiredIn::test_response_body_is_extracted_not_returned_raw` —
  replaces the coverage the 7 moved tests used to give incidentally, proving
  `fetch_url` still routes the body through the extraction seam.
- `TestExtractionSeamIsPure` (2) — assert `web_html` imports no `urllib`/
  `socket` and does not import `web`, so the split cannot silently reverse.
- `TestHtmlExtraction::test_text_free_html_yields_empty_string` — pins the
  contract at the new seam: the pure module *reports* emptiness, `web.py`
  decides it is an error.
- `TestHtmlExtraction::test_link_text_is_kept_but_href_is_not` — the pure
  counterpart of the same assertion in `TestNoRecursion`, which stays in
  transport because there it means "we did not fetch the link".

Two files assert the link-text behavior (once pure, once as part of the
no-recursion guarantee). That is deliberate: same assertion, different claim.

## Second structural split — transport and body reading

No behavior change. After the HTML split, `web.py` was still 470 lines holding
three unrelated jobs: fetch policy, urllib plumbing, and stream reading.

### Baseline (before)

```bash
$ python3 -m pytest -q
227 passed in 5.48s
```

### After

```bash
$ python3 -m pytest tests/voice_studio/test_web_transport.py -q
76 passed in 0.24s
$ python3 -m pytest -q
227 passed in 5.39s
```

Same 227 tests, same names — this split moved code, not coverage.

### Mapping

| Moved | To | Contents |
|---|---|---|
| `_NoRedirect`, `_pinned_connection_factory`, `_PinnedHTTPHandler`, `_PinnedHTTPSHandler`, `_build_opener`, `_default_opener`, `USER_AGENT` | `web_transport.py` (128 lines) | urllib wiring, verbatim |
| `_apply_read_timeout`, `_ReadDeadlineExceeded`, `_read_bounded`, `READ_CHUNK_BYTES` | `web_body.py` (87 lines) | bounded deadline-aware reading, verbatim |
| — | `web.py` (470 → 275 lines) | `is_url`, `_validate_url`, `_default_resolver`, `_is_blocked_address`, `_validate_destination`, `_content_type`, `fetch_url`, and the limit constants |

Module sizes now: `web.py` 275, `web_transport.py` 128, `web_body.py` 87,
`web_html.py` 79.

### Security invariants preserved

Verified by the unchanged tests, not by inspection: redirect refusal (8 tests),
address pinning with Host/SNI intact (7), destination block list (20), the
per-read socket deadline (5), byte cap, and content-type allowlist. The
validate-then-pin sequence still lives in `fetch_url` — `web_transport` dials
the address it is given and performs no validation of its own, so policy has
exactly one home.

### Two adjustments

- `web.py` imports only what it uses (`_default_opener`, `_read_bounded`,
  `_ReadDeadlineExceeded`, `extract_text`). An earlier cut re-exported
  `_NoRedirect`, `_build_opener`, `_pinned_connection_factory`, and
  `_apply_read_timeout` purely so tests could reach them through `web`; that is
  a second home for a name, so the re-exports were dropped.
- Consequently seven test references now point at the real module
  (`web_transport._NoRedirect`, `web_body._apply_read_timeout`, …). Test names
  and assertions are unchanged; only the module path moved.

A check confirms the one-way dependency: none of `web_html`, `web_body`, or
`web_transport` imports `web`.

## Third structural split — test support and security tests

No behavior change, no production code touched. `test_web_transport.py` was
764 lines mixing shared fakes, transport behavior, and the security guarantees.

### Baseline (before)

```bash
$ python3 -m pytest -q
227 passed in 5.55s
$ python3 -m pytest tests/voice_studio/test_web_transport.py -q
76 passed
```

The 76 test IDs were captured before editing and diffed afterwards.

### After

```bash
$ python3 -m pytest tests/voice_studio/test_web_transport.py tests/voice_studio/test_web_security.py -q
76 passed in 0.24s
$ python3 -m pytest -q
227 passed in 5.59s
$ git diff --check
(clean)
```

Test-ID diff: **empty in both directions** — not one name added, removed, or
renamed. File sizes: `test_web_transport.py` 320, `test_web_security.py` 400,
`web_test_support.py` 74.

### Mapping

| Moved | To |
|---|---|
| `FakeHeaders`, `FakeResponse`, `opener_of`, `PAGE`, `no_live_dns` | `web_test_support.py`, verbatim |
| `TestDestinationValidation`, `TestReadDeadline`, `TestPerReadTimeout`, `TestAddressPinning` | `test_web_security.py`, verbatim |
| `TestUrlValidation`, `TestExtractionIsWiredIn`, `TestResponseLimits`, `TestContentType`, `TestTransportErrors`, `TestNoRecursion`, `TestRedirects`, `TestDispatch` | stayed in `test_web_transport.py` |

### The one thing that could have broken silently

`no_live_dns` is an **autouse** fixture. Autouse applies only where the fixture
is visible, so moving it to a support module could have left both files
reaching real DNS while every test still passed — the failure mode would have
been invisible in a green run. Both modules therefore import it explicitly
(`# noqa: F401` — it is used implicitly, not by name), and this was verified
rather than assumed:

```bash
$ python3 -m pytest <one test from each file> --fixtures-per-test -q | grep no_live_dns
no_live_dns -- tests/voice_studio/web_test_support.py:41
no_live_dns -- tests/voice_studio/web_test_support.py:41
```

Both files show the fixture attached. A grep also confirms no test calls
`socket.getaddrinfo` or `urlopen`.

`test_web_transport.py` additionally imports `web_transport`, which
`TestRedirects` needs for `_NoRedirect` and `_build_opener` — caught by two
failing tests immediately after the split, fixed by the import, no assertion
changed.

## Fourth structural split — pinning tests

No behavior change, no production code touched. `test_web_security.py` covered
two distinct claims: "we refuse to talk to internal addresses" and "we dial the
address we validated".

### Baseline (before)

```bash
$ python3 -m pytest -q
227 passed in 5.52s
$ python3 -m pytest tests/voice_studio/test_web_security.py -q
39 passed
```

The 39 test IDs were captured before editing and diffed afterwards.

### After

```bash
$ python3 -m pytest tests/voice_studio/test_web_security.py -q
32 passed in 0.08s
$ python3 -m pytest tests/voice_studio/test_web_pinning.py -q
7 passed in 0.10s
$ python3 -m pytest -q
227 passed in 5.55s
$ git diff --check
(clean)
```

Test-ID diff: **empty in both directions**. 32 + 7 = 39. File sizes:
`test_web_security.py` 312, `test_web_pinning.py` 102.

### Mapping

| Class | Destination |
|---|---|
| `TestAddressPinning` (7) | `test_web_pinning.py`, verbatim |
| `TestDestinationValidation` (20), `TestReadDeadline` (5), `TestPerReadTimeout` (5) | stayed in `test_web_security.py` |

`test_web_security.py` no longer imports `web_transport` — only
`TestAddressPinning` used it — so each file imports exactly what it needs.
`test_web_pinning.py` imports `FakeResponse`, `PAGE`, and the autouse
`no_live_dns` from `web_test_support`; it needs neither `FakeHeaders` nor
`opener_of`.

The autouse DNS guard was verified attached in the new file rather than
assumed:

```bash
$ python3 -m pytest tests/voice_studio/test_web_pinning.py::TestAddressPinning::test_ip_literal_is_pinned_to_itself --fixtures-per-test -q | grep no_live_dns
no_live_dns -- tests/voice_studio/web_test_support.py:41
```

This matters most here: the pinning tests are the ones that would otherwise
resolve hostnames for real.

## Design notes

- `is_url()` is checked in `parse_source_file` **before** `Path()` conversion,
  so a URL is never reported as a missing file.
- The body is read as `read(MAX_WEB_RESPONSE_BYTES + 1)` — one byte past the
  cap — so an oversized response is detected without ever materializing it.
- `HTTPError` is caught before the general handler because it is a subclass of
  `URLError`; catching it first is what preserves the status code in the message.
- No link following, no retry, no second request under any condition. The
  no-burst guarantee is structural and is asserted by `TestNoRecursion`.
- Boilerplate (nav, cookie banners, footers) is **not** stripped. That is a
  known limitation, recorded in the ingestion CONTEXT human check rather than
  papered over with a heuristic.
