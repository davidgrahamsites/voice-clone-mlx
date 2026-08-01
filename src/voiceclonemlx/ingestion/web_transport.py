"""Open one URL over HTTP(S) with redirects disabled and the address pinned.

Owns the urllib wiring only: connection classes, handlers, opener assembly.
It applies transport policy decided elsewhere — `web.py` validates the URL and
the destination address, then hands the approved address here to dial.
"""

import http.client
import socket
import urllib.request

USER_AGENT = "VoiceCloneMLX/1.0 (script ingestion)"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects, so one call means one request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _pinned_connection_factory(pinned_ip: str, secure: bool, context=None):
    """Build a connection class that dials one fixed address.

    The hostname is left untouched on the connection, so the `Host` header and
    the TLS SNI name still identify the site; only the address we dial is
    pinned. This closes the window between validating a resolved address and
    connecting: a second DNS lookup cannot redirect the connection.

    Args:
        pinned_ip: The already-validated address to connect to
        secure: True for HTTPS
        context: Optional SSL context for HTTPS

    Returns:
        A callable with the signature urllib expects for a connection class
    """
    base = http.client.HTTPSConnection if secure else http.client.HTTPConnection

    class _PinnedConnection(base):
        _pinned_ip = pinned_ip

        def connect(self):
            sock = socket.create_connection(
                (self._pinned_ip, self.port),
                self.timeout,
                self.source_address,
            )

            if self._tunnel_host:
                self.sock = sock
                self._tunnel()
                sock = self.sock

            if secure:
                # server_hostname stays the real hostname: pinning must not
                # weaken certificate or SNI verification.
                self.sock = self._context.wrap_socket(
                    sock, server_hostname=self.host
                )
            else:
                self.sock = sock

    def factory(host, **kwargs):
        if secure and context is not None:
            kwargs.setdefault("context", context)
        return _PinnedConnection(host, **kwargs)

    return factory


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    """HTTP handler that dials a pinned address."""

    def __init__(self, pinned_ip):
        super().__init__()
        self._pinned_ip = pinned_ip

    def http_open(self, req):
        return self.do_open(
            _pinned_connection_factory(self._pinned_ip, secure=False), req
        )


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that dials a pinned address, keeping SNI intact."""

    def __init__(self, pinned_ip):
        super().__init__()
        self._pinned_ip = pinned_ip

    def https_open(self, req):
        return self.do_open(
            _pinned_connection_factory(
                self._pinned_ip, secure=True, context=self._context
            ),
            req,
        )


def _build_opener(pinned_ip: str = None):
    """Build a urllib opener with redirects disabled and the address pinned.

    Args:
        pinned_ip: Validated address to dial, or None to use normal resolution
    """
    handlers = [_NoRedirect]

    if pinned_ip:
        handlers.append(_PinnedHTTPHandler(pinned_ip))
        handlers.append(_PinnedHTTPSHandler(pinned_ip))

    return urllib.request.build_opener(*handlers)


def _default_opener(url: str, timeout: int, pinned_ip: str = None):
    """Open a URL with urllib, without following redirects.

    Args:
        url: Validated http(s) URL
        timeout: Socket timeout in seconds
        pinned_ip: Validated address to connect to

    Returns:
        The response object from urllib
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return _build_opener(pinned_ip).open(request, timeout=timeout)
