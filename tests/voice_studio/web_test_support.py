"""Shared fakes for the website ingestion tests.

Fetching is faked here so no test opens a socket or resolves a hostname. Import
`no_live_dns` alongside the fakes: it is autouse, so importing it into a test
module arms it for every test in that module.
"""

import pytest

from voiceclonegpt.ingestion import web


class FakeHeaders(dict):
    """Stands in for http.client.HTTPMessage."""


class FakeResponse:
    """Stands in for the object urlopen returns.

    Reads consume the body, so chunked reading terminates like a real stream.
    """

    def __init__(self, body, content_type="text/html; charset=utf-8", status=200):
        self._body = body if isinstance(body, bytes) else body.encode("utf-8")
        self._offset = 0
        self.status = status
        self.headers = FakeHeaders({"Content-Type": content_type})

    def read(self, size=None):
        if size is None:
            chunk = self._body[self._offset:]
            self._offset = len(self._body)
            return chunk

        chunk = self._body[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


@pytest.fixture(autouse=True)
def no_live_dns(monkeypatch):
    """No test may resolve a hostname for real.

    Every host used in this module resolves to a public address unless the
    test overrides it.
    """

    def offline_resolver(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(web, "_default_resolver", offline_resolver)


def opener_of(body, content_type="text/html; charset=utf-8"):
    """Build an opener returning a fixed response."""
    return lambda url, timeout: FakeResponse(body, content_type)


PAGE = """
<html>
  <head>
    <title>Ignored title</title>
    <style>body { color: red; }</style>
    <script>alert('hostile');</script>
  </head>
  <body>
    <h1>Real Heading</h1>
    <p>First paragraph of readable text.</p>
    <script>tracker.send('nope');</script>
    <p>Second paragraph.</p>
    <noscript>Enable JavaScript.</noscript>
  </body>
</html>
"""
