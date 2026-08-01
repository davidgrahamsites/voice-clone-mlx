"""Test website fetching: URL policy, transport wiring, limits, and dispatch.

HTML extraction lives in `test_web_html.py`; destination and resource-limit
security lives in `test_web_security.py`. No test here touches the network.
"""

import pytest

from voiceclonegpt.ingestion import web, web_transport
from voiceclonegpt.ingestion.parsers import parse_source_file
from web_test_support import (  # noqa: F401  (no_live_dns is autouse)
    FakeHeaders,
    FakeResponse,
    PAGE,
    no_live_dns,
    opener_of,
)


class TestUrlValidation:
    """Only http and https may be fetched."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/doc.txt",
            "javascript:alert(1)",
            "data:text/html,<h1>hi</h1>",
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            web.fetch_url(url, opener=opener_of(PAGE))

    def test_rejects_url_without_host(self):
        with pytest.raises(ValueError, match="no host"):
            web.fetch_url("http://", opener=opener_of(PAGE))

    def test_rejects_scheme_case_insensitively(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            web.fetch_url("FILE:///etc/passwd", opener=opener_of(PAGE))

    def test_accepts_https(self):
        text = web.fetch_url("https://example.com", opener=opener_of(PAGE))
        assert "First paragraph" in text

    def test_validation_happens_before_any_fetch(self):
        """A rejected URL must never reach the opener."""
        calls = []

        def recording_opener(url, timeout):
            calls.append(url)
            return FakeResponse(PAGE)

        with pytest.raises(ValueError):
            web.fetch_url("file:///etc/passwd", opener=recording_opener)

        assert calls == []


class TestExtractionIsWiredIn:
    """fetch_url runs the response body through the HTML seam."""

    def test_response_body_is_extracted_not_returned_raw(self):
        text = web.fetch_url("https://example.com", opener=opener_of(PAGE))

        assert "Real Heading" in text
        assert "<h1>" not in text

    def test_rejects_page_with_no_readable_text(self):
        html = "<html><head><script>x=1;</script></head><body></body></html>"

        with pytest.raises(ValueError, match="no readable text"):
            web.fetch_url("https://example.com", opener=opener_of(html))


class TestResponseLimits:
    """Bounded reads, bounded text, bounded time."""

    def test_rejects_oversized_response(self, monkeypatch):
        monkeypatch.setattr(web, "MAX_WEB_RESPONSE_BYTES", 50)
        html = "<p>" + ("x" * 500) + "</p>"

        with pytest.raises(ValueError, match="Response too large"):
            web.fetch_url("https://example.com", opener=opener_of(html))

    def test_reads_a_bounded_number_of_bytes(self):
        """The adapter must not call read() unbounded."""
        seen = []

        class RecordingResponse(FakeResponse):
            def read(self, size=None):
                seen.append(size)
                return super().read(size)

        web.fetch_url(
            "https://example.com",
            opener=lambda url, timeout: RecordingResponse(PAGE),
        )

        assert seen and all(s is not None for s in seen)

    def test_rejects_oversized_extracted_text(self, monkeypatch):
        monkeypatch.setattr(web, "MAX_WEB_TEXT_CHARS", 10)
        html = "<p>" + ("word " * 100) + "</p>"

        with pytest.raises(ValueError, match="extracted text too large"):
            web.fetch_url("https://example.com", opener=opener_of(html))

    def test_passes_the_configured_timeout(self):
        seen = {}

        def recording_opener(url, timeout):
            seen["timeout"] = timeout
            return FakeResponse(PAGE)

        web.fetch_url("https://example.com", opener=recording_opener)

        assert seen["timeout"] == web.WEB_TIMEOUT_SECONDS

    def test_timeout_is_overridable(self):
        seen = {}

        def recording_opener(url, timeout):
            seen["timeout"] = timeout
            return FakeResponse(PAGE)

        web.fetch_url("https://example.com", opener=recording_opener, timeout=3)

        assert seen["timeout"] == 3


class TestContentType:
    """Only textual responses are accepted."""

    def test_accepts_plain_text(self):
        text = web.fetch_url(
            "https://example.com",
            opener=opener_of("Just words.", content_type="text/plain"),
        )

        assert "Just words." in text

    def test_rejects_binary_content_type(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            web.fetch_url(
                "https://example.com",
                opener=opener_of(b"%PDF-1.4", content_type="application/pdf"),
            )

    def test_missing_content_type_is_treated_as_html(self):
        text = web.fetch_url(
            "https://example.com", opener=opener_of(PAGE, content_type="")
        )

        assert "Real Heading" in text


class TestTransportErrors:
    """Network and HTTP failures become stable, chained ValueErrors."""

    def test_http_error_becomes_value_error(self):
        import urllib.error

        def failing_opener(url, timeout):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with pytest.raises(ValueError, match="HTTP 404"):
            web.fetch_url("https://example.com", opener=failing_opener)

    def test_url_error_becomes_value_error(self):
        import urllib.error

        def failing_opener(url, timeout):
            raise urllib.error.URLError("name resolution failed")

        with pytest.raises(ValueError, match="Could not fetch"):
            web.fetch_url("https://example.com", opener=failing_opener)

    def test_transport_error_preserves_cause(self):
        import urllib.error

        def failing_opener(url, timeout):
            raise urllib.error.URLError("name resolution failed")

        with pytest.raises(ValueError) as excinfo:
            web.fetch_url("https://example.com", opener=failing_opener)

        assert isinstance(excinfo.value.__cause__, urllib.error.URLError)

    def test_timeout_becomes_value_error(self):
        def slow_opener(url, timeout):
            raise TimeoutError("timed out")

        with pytest.raises(ValueError, match="Could not fetch"):
            web.fetch_url("https://example.com", opener=slow_opener)

    def test_adapter_errors_are_not_normalized(self):
        """Our own limit errors keep their message and carry no cause."""
        html = "<html><body></body></html>"

        with pytest.raises(ValueError) as excinfo:
            web.fetch_url("https://example.com", opener=opener_of(html))

        assert "no readable text" in str(excinfo.value)
        assert excinfo.value.__cause__ is None


class TestNoRecursion:
    """The adapter fetches exactly one URL."""

    def test_links_are_not_followed(self):
        calls = []

        def counting_opener(url, timeout):
            calls.append(url)
            return FakeResponse(
                '<p>Text.</p><a href="https://example.com/next">Next</a>'
            )

        web.fetch_url("https://example.com", opener=counting_opener)

        assert calls == ["https://example.com"]

    def test_link_text_is_kept_but_href_is_not(self):
        text = web.fetch_url(
            "https://example.com",
            opener=opener_of('<p>See <a href="https://evil.test">this</a>.</p>'),
        )

        assert "this" in text
        assert "evil.test" not in text


class TestRedirects:
    """Exactly one request: redirects are rejected, never followed."""

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_rejects_redirect_status(self, status):
        response = FakeResponse(PAGE, status=status)

        with pytest.raises(ValueError, match="Redirect"):
            web.fetch_url(
                "https://example.com", opener=lambda url, timeout: response
            )

    def test_redirect_http_error_is_reported_as_a_redirect(self):
        """urllib raises HTTPError for 3xx once redirects are disabled."""
        import urllib.error

        def redirecting_opener(url, timeout):
            raise urllib.error.HTTPError(
                url, 302, "Found", {"Location": "https://elsewhere.test"}, None
            )

        with pytest.raises(ValueError, match="Redirect"):
            web.fetch_url("https://example.com", opener=redirecting_opener)

    def test_redirect_body_is_never_read(self):
        """A 3xx response must be rejected before its body is touched."""
        reads = []

        class RecordingResponse(FakeResponse):
            def read(self, size=None):
                reads.append(size)
                return super().read(size)

        response = RecordingResponse(PAGE, status=302)

        with pytest.raises(ValueError, match="Redirect"):
            web.fetch_url(
                "https://example.com", opener=lambda url, timeout: response
            )

        assert reads == []

    def test_default_opener_has_redirects_disabled(self):
        """The urllib opener is built with redirect following removed."""
        handler = web_transport._NoRedirect()

        assert handler.redirect_request(None, None, 302, "Found", {}, "u") is None

    def test_default_opener_is_not_the_global_urlopen(self):
        """A custom opener is built rather than using urllib's default."""
        opener = web_transport._build_opener()

        assert any(
            isinstance(h, web_transport._NoRedirect) for h in opener.handlers
        )


class TestDispatch:
    """parse_source_file routes URLs to the web adapter."""

    def test_http_url_is_dispatched(self, monkeypatch):
        monkeypatch.setattr(
            web,
            "_default_opener",
            lambda url, timeout, pinned_ip=None: FakeResponse(PAGE),
        )

        text = parse_source_file("https://example.com")
        assert "Real Heading" in text

    def test_url_is_not_treated_as_a_missing_file(self, monkeypatch):
        monkeypatch.setattr(
            web,
            "_default_opener",
            lambda url, timeout, pinned_ip=None: FakeResponse(PAGE),
        )

        # Must not raise FileNotFoundError
        assert parse_source_file("http://example.com/page.html")

    def test_non_url_paths_still_work(self, tmp_path):
        path = tmp_path / "source.txt"
        path.write_text("Local text.")

        assert parse_source_file(path) == "Local text."
