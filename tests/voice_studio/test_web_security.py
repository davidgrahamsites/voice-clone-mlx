"""Test the website adapter's security guarantees.

Destination allowlisting (SSRF), the read deadline, and per-read socket
timeouts. Address pinning lives in `test_web_pinning.py`. No test here touches
the network or resolves a hostname.
"""

import pytest

from voiceclonegpt.ingestion import web, web_body
from web_test_support import (  # noqa: F401  (no_live_dns is autouse)
    FakeHeaders,
    FakeResponse,
    PAGE,
    no_live_dns,
    opener_of,
)


class TestDestinationValidation:
    """Internal and special-use addresses are never fetched."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",        # loopback
            "127.9.9.9",        # loopback range
            "10.0.0.5",         # private
            "192.168.1.1",      # private
            "172.16.0.1",       # private
            "169.254.169.254",  # link-local / cloud metadata
            "0.0.0.0",          # unspecified
            "224.0.0.1",        # multicast
            "[::1]",            # IPv6 loopback
            "[fe80::1]",        # IPv6 link-local
            "[fc00::1]",        # IPv6 unique-local
            "[::]",             # IPv6 unspecified
            "[::ffff:127.0.0.1]",  # IPv4-mapped loopback
        ],
    )
    def test_rejects_internal_ip_literals(self, host):
        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url(f"http://{host}/page", opener=opener_of(PAGE))

    def test_allows_public_ip_literal(self):
        text = web.fetch_url("http://93.184.216.34/", opener=opener_of(PAGE))

        assert "Real Heading" in text

    def test_rejects_hostname_resolving_to_loopback(self, monkeypatch):
        monkeypatch.setattr(web, "_default_resolver", lambda host: ["127.0.0.1"])

        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url("http://localhost/page", opener=opener_of(PAGE))

    def test_rejects_hostname_resolving_to_metadata_address(self, monkeypatch):
        monkeypatch.setattr(
            web, "_default_resolver", lambda host: ["169.254.169.254"]
        )

        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url("http://metadata.test/", opener=opener_of(PAGE))

    def test_rejects_when_any_resolved_address_is_internal(self, monkeypatch):
        """A host resolving to both public and private addresses is rejected."""
        monkeypatch.setattr(
            web,
            "_default_resolver",
            lambda host: ["93.184.216.34", "10.0.0.5"],
        )

        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url("http://mixed.test/", opener=opener_of(PAGE))

    def test_allows_hostname_resolving_to_public_address(self, monkeypatch):
        monkeypatch.setattr(
            web, "_default_resolver", lambda host: ["93.184.216.34"]
        )

        assert web.fetch_url("http://example.com/", opener=opener_of(PAGE))

    def test_rejects_unresolvable_host(self, monkeypatch):
        def failing_resolver(host):
            raise OSError("Name or service not known")

        monkeypatch.setattr(web, "_default_resolver", failing_resolver)

        with pytest.raises(ValueError, match="Could not resolve"):
            web.fetch_url("http://nope.test/", opener=opener_of(PAGE))

    def test_rejects_host_resolving_to_nothing(self, monkeypatch):
        monkeypatch.setattr(web, "_default_resolver", lambda host: [])

        with pytest.raises(ValueError, match="Could not resolve"):
            web.fetch_url("http://empty.test/", opener=opener_of(PAGE))

    def test_destination_is_validated_before_any_fetch(self):
        calls = []

        def recording_opener(url, timeout):
            calls.append(url)
            return FakeResponse(PAGE)

        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url("http://127.0.0.1/", opener=recording_opener)

        assert calls == []

    def test_resolver_is_injectable(self):
        """The resolver seam is a parameter, not only a module attribute."""
        with pytest.raises(ValueError, match="not allowed"):
            web.fetch_url(
                "http://anything.test/",
                opener=opener_of(PAGE),
                resolver=lambda host: ["192.168.0.9"],
            )


class TestReadDeadline:
    """The body read is bounded in time, not only in bytes."""

    def test_read_that_outlasts_the_deadline_is_rejected(self):
        """A response that dribbles bytes forever is abandoned."""
        clock = iter(range(0, 10_000, 5))  # 5 fake seconds per tick

        class SlowResponse:
            status = 200
            headers = FakeHeaders({"Content-Type": "text/html"})

            def read(self, size=None):
                return b"x" * 10  # never reaches EOF

            def close(self):
                pass

        with pytest.raises(ValueError, match="Timed out reading"):
            web.fetch_url(
                "https://example.com",
                opener=lambda url, timeout: SlowResponse(),
                timeout=10,
                clock=lambda: next(clock),
            )

    def test_read_within_the_deadline_succeeds(self):
        ticks = iter([0, 0, 1, 1, 2, 2, 3, 3])

        text = web.fetch_url(
            "https://example.com",
            opener=opener_of(PAGE),
            timeout=30,
            clock=lambda: next(ticks),
        )

        assert "Real Heading" in text

    def test_deadline_is_derived_from_the_timeout(self):
        """A slow read is cut off at the timeout, not at some fixed value."""
        clock = iter(range(0, 10_000, 1))

        class SlowResponse:
            status = 200
            headers = FakeHeaders({"Content-Type": "text/html"})

            def read(self, size=None):
                return b"x" * 10

            def close(self):
                pass

        with pytest.raises(ValueError, match="Timed out reading"):
            web.fetch_url(
                "https://example.com",
                opener=lambda url, timeout: SlowResponse(),
                timeout=2,
                clock=lambda: next(clock),
            )

    def test_transport_value_error_during_read_is_normalized(self):
        """A ValueError from the transport is not mistaken for our deadline."""

        class ExplodingResponse:
            status = 200
            headers = FakeHeaders({"Content-Type": "text/html"})

            def read(self, size=None):
                raise ValueError("chunked encoding is broken")

        with pytest.raises(ValueError, match="Could not fetch") as excinfo:
            web.fetch_url(
                "https://example.com",
                opener=lambda url, timeout: ExplodingResponse(),
            )

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_oversized_body_still_rejected_by_bytes(self, monkeypatch):
        """The byte cap still applies when the deadline is generous."""
        monkeypatch.setattr(web, "MAX_WEB_RESPONSE_BYTES", 50)

        with pytest.raises(ValueError, match="Response too large"):
            web.fetch_url(
                "https://example.com",
                opener=opener_of("<p>" + "x" * 500 + "</p>"),
                timeout=600,
            )


class TestPerReadTimeout:
    """The deadline must also bound a single blocking read."""

    def test_socket_timeout_is_set_before_each_read(self):
        applied = []

        def recording_setter(response, seconds):
            applied.append(seconds)

        ticks = iter([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])

        web.fetch_url(
            "https://example.com",
            opener=opener_of(PAGE),
            timeout=30,
            clock=lambda: next(ticks),
            read_timeout_setter=recording_setter,
        )

        assert applied, "no read timeout was applied"

    def test_applied_timeout_shrinks_as_the_deadline_approaches(self):
        applied = []

        def recording_setter(response, seconds):
            applied.append(seconds)

        clock = iter([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        class TwoChunkResponse:
            status = 200
            headers = FakeHeaders({"Content-Type": "text/html"})

            def __init__(self):
                self._chunks = [b"<p>One.</p>", b"<p>Two.</p>", b""]

            def read(self, size=None):
                return self._chunks.pop(0) if self._chunks else b""

        web.fetch_url(
            "https://example.com",
            opener=lambda url, timeout: TwoChunkResponse(),
            timeout=20,
            clock=lambda: next(clock),
            read_timeout_setter=recording_setter,
        )

        assert len(applied) >= 2
        assert applied == sorted(applied, reverse=True)
        assert all(value > 0 for value in applied)

    def test_single_blocking_read_cannot_outlast_the_deadline(self):
        """A read that blocks past the deadline is cut off by the socket."""
        applied = []

        def recording_setter(response, seconds):
            applied.append(seconds)

        clock = iter([0, 0, 19, 19, 25])

        class BlockingResponse:
            status = 200
            headers = FakeHeaders({"Content-Type": "text/html"})

            def read(self, size=None):
                # Honours whatever timeout the adapter last applied.
                if applied and applied[-1] <= 1:
                    raise TimeoutError("timed out")
                return b"x" * 10

        with pytest.raises(ValueError, match="Could not fetch|Timed out"):
            web.fetch_url(
                "https://example.com",
                opener=lambda url, timeout: BlockingResponse(),
                timeout=20,
                clock=lambda: next(clock),
                read_timeout_setter=recording_setter,
            )

        assert applied[-1] <= 1

    def test_default_setter_tolerates_a_response_without_a_socket(self):
        """Fakes and exotic responses must not break the adapter."""
        web_body._apply_read_timeout(FakeResponse(PAGE), 5)  # must not raise

    def test_default_setter_sets_the_underlying_socket_timeout(self):
        """The real urllib response shape is reached through fp.raw._sock."""
        seen = []

        class Sock:
            def settimeout(self, value):
                seen.append(value)

        class Raw:
            _sock = Sock()

        class Fp:
            raw = Raw()

        class Response:
            fp = Fp()

        web_body._apply_read_timeout(Response(), 7.5)

        assert seen == [7.5]
