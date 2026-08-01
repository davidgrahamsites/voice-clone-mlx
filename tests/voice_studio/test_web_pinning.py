"""Test DNS-rebinding-proof address pinning.

The validated address is the one dialed, and pinning must not weaken Host or
SNI. No test here touches the network or resolves a hostname.
"""

import pytest

from voiceclonemlx.ingestion import web, web_transport
from web_test_support import (  # noqa: F401  (no_live_dns is autouse)
    FakeResponse,
    PAGE,
    no_live_dns,
)


class TestAddressPinning:
    """The connection goes to the address that was validated."""

    def test_validated_address_is_pinned_for_the_connection(self, monkeypatch):
        seen = {}

        def recording_default_opener(url, timeout, pinned_ip=None):
            seen["pinned_ip"] = pinned_ip
            return FakeResponse(PAGE)

        monkeypatch.setattr(web, "_default_opener", recording_default_opener)
        monkeypatch.setattr(
            web, "_default_resolver", lambda host: ["93.184.216.34"]
        )

        web.fetch_url("http://example.com/")

        assert seen["pinned_ip"] == "93.184.216.34"

    def test_second_resolution_cannot_bypass_the_block(self, monkeypatch):
        """DNS rebinding: a later lookup returning a private IP is never used."""
        resolutions = []

        def rebinding_resolver(host):
            resolutions.append(host)
            # First lookup is public; every later lookup is internal.
            if len(resolutions) == 1:
                return ["93.184.216.34"]
            return ["127.0.0.1"]

        seen = {}

        def recording_default_opener(url, timeout, pinned_ip=None):
            seen["pinned_ip"] = pinned_ip
            return FakeResponse(PAGE)

        monkeypatch.setattr(web, "_default_opener", recording_default_opener)
        monkeypatch.setattr(web, "_default_resolver", rebinding_resolver)

        web.fetch_url("http://example.com/")

        # Resolved once, and the connection used that validated address.
        assert len(resolutions) == 1
        assert seen["pinned_ip"] == "93.184.216.34"

    def test_ip_literal_is_pinned_to_itself(self, monkeypatch):
        seen = {}

        def recording_default_opener(url, timeout, pinned_ip=None):
            seen["pinned_ip"] = pinned_ip
            return FakeResponse(PAGE)

        monkeypatch.setattr(web, "_default_opener", recording_default_opener)

        web.fetch_url("http://93.184.216.34/")

        assert seen["pinned_ip"] == "93.184.216.34"

    def test_validate_destination_returns_the_address_it_approved(self):
        approved = web._validate_destination(
            "http://example.com/", resolver=lambda host: ["93.184.216.34"]
        )

        assert approved == "93.184.216.34"

    def test_pinned_connection_keeps_the_hostname_for_host_and_sni(self):
        """Pinning changes where we connect, not who we claim to be."""
        connection = web_transport._pinned_connection_factory(
            "93.184.216.34", secure=False
        )("example.com", timeout=5)

        assert connection.host == "example.com"
        assert connection._pinned_ip == "93.184.216.34"

    def test_pinned_https_connection_keeps_the_hostname(self):
        connection = web_transport._pinned_connection_factory(
            "93.184.216.34", secure=True
        )("example.com", timeout=5)

        assert connection.host == "example.com"
        assert connection._pinned_ip == "93.184.216.34"

    def test_pinned_opener_is_still_redirect_free(self):
        opener = web_transport._build_opener(pinned_ip="93.184.216.34")

        assert any(isinstance(h, web_transport._NoRedirect) for h in opener.handlers)
