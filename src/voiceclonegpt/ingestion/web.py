"""Fetch a single web page and extract its readable text.

Deliberately narrow: one URL, one request, no link following, no crawling.
The network call is behind an `opener` seam so every rule here is testable
without touching a live service.
"""

import ipaddress
import socket
import time
import urllib.error
from urllib.parse import urlparse

from voiceclonegpt.ingestion.web_body import (
    _ReadDeadlineExceeded,
    _read_bounded,
)
from voiceclonegpt.ingestion.web_html import extract_text
from voiceclonegpt.ingestion.web_transport import _default_opener

ALLOWED_SCHEMES = ("http", "https")
WEB_TIMEOUT_SECONDS = 10
MAX_WEB_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_WEB_TEXT_CHARS = 5_000_000

TEXTUAL_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")


def is_url(candidate) -> bool:
    """Return True if the value looks like an http(s) URL.

    Used by the dispatcher to tell a URL from a filesystem path.
    """
    return str(candidate).lower().startswith(("http://", "https://"))


def _validate_url(url: str) -> None:
    """Reject anything that is not a fetchable http(s) URL.

    Raises:
        ValueError: If the scheme is not http/https or no host is present
    """
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme or '(none)'}. "
            f"Supported: http, https"
        )

    if not parsed.netloc:
        raise ValueError(f"URL has no host: {url}")


def _default_resolver(host: str):
    """Resolve a hostname to its IP addresses.

    Args:
        host: Hostname to resolve

    Returns:
        List of IP address strings

    Raises:
        OSError: If resolution fails
    """
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _is_blocked_address(ip) -> bool:
    """Return True for addresses that must never be fetched.

    Blocks loopback, private, link-local (including cloud metadata at
    169.254.169.254), multicast, unspecified, and reserved ranges, plus
    IPv4-mapped IPv6 forms of the same.
    """
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _validate_destination(url: str, resolver=None) -> str:
    """Reject URLs pointing at internal or special-use addresses.

    An IP literal is checked directly. A hostname is resolved **once** and
    every returned address must be allowed — a host resolving to both a public
    and a private address is rejected. The approved address is returned so the
    caller can dial exactly what was validated; nothing resolves the host a
    second time.

    Args:
        url: Validated http(s) URL
        resolver: Callable `(host) -> [ip strings]`. Defaults to
            `_default_resolver`; injectable so tests never touch DNS.

    Returns:
        The validated address to connect to

    Raises:
        ValueError: If any destination address is blocked, or the host cannot
            be resolved
    """
    host = urlparse(url).hostname or ""

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        if _is_blocked_address(literal):
            raise ValueError(
                f"Destination address is not allowed: {host}. "
                f"Loopback, private, link-local, multicast, and reserved "
                f"addresses cannot be fetched."
            )
        return host

    if resolver is None:
        resolver = _default_resolver

    try:
        addresses = list(resolver(host))
    except Exception as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc

    if not addresses:
        raise ValueError(f"Could not resolve host: {host}")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise ValueError(f"Could not resolve host: {host}")

        if _is_blocked_address(ip):
            raise ValueError(
                f"Destination address is not allowed: {host} resolves to "
                f"{address}. Loopback, private, link-local, multicast, and "
                f"reserved addresses cannot be fetched."
            )

    return addresses[0]


def _content_type(response) -> str:
    """Read the Content-Type header, tolerating responses without one."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""

    return (headers.get("Content-Type") or "").split(";")[0].strip().lower()


def fetch_url(
    url: str,
    opener=None,
    timeout: int = None,
    resolver=None,
    clock=None,
    read_timeout_setter=None,
) -> str:
    """Fetch one web page and return its readable text.

    Fetches exactly the URL given: redirects are rejected rather than
    followed, links are never followed, and no second request is made.

    Args:
        url: An http or https URL
        opener: Callable `(url, timeout) -> response`. Defaults to
            `_default_opener` (urllib, redirects disabled); injectable.
        timeout: Socket timeout and read deadline in seconds. Defaults to
            WEB_TIMEOUT_SECONDS.
        resolver: Callable `(host) -> [ip strings]`. Defaults to
            `_default_resolver`; injectable so tests never touch DNS.
        clock: Callable returning the current time, for the read deadline.
            Defaults to `time.monotonic`.
        read_timeout_setter: Callable `(response, seconds)` applying the
            remaining deadline to the socket before each read. Defaults to
            `_apply_read_timeout`; injectable for tests.

    Returns:
        Readable page text

    Raises:
        ValueError: Bad scheme or host, blocked destination address,
            unresolvable host, redirect response, transport or HTTP failure,
            read deadline exceeded, unsupported content type, response or
            extracted text over the limit, or a page with no readable text
    """
    _validate_url(url)
    # Resolved once; the same address is dialed, so no second lookup can
    # redirect the connection after validation.
    pinned_ip = _validate_destination(url, resolver)

    if opener is None:
        def opener(request_url, request_timeout):
            return _default_opener(
                request_url, request_timeout, pinned_ip=pinned_ip
            )
    if timeout is None:
        timeout = WEB_TIMEOUT_SECONDS
    if clock is None:
        clock = time.monotonic

    try:
        response = opener(url, timeout)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise ValueError(
                f"Redirect not followed: {url} returned HTTP {exc.code}. "
                f"Supply the final URL directly."
            ) from exc
        raise ValueError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except Exception as exc:
        raise ValueError(f"Could not fetch {url}: {exc}") from exc

    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    if status is not None and 300 <= status < 400:
        raise ValueError(
            f"Redirect not followed: {url} returned HTTP {status}. "
            f"Supply the final URL directly."
        )

    deadline = clock() + timeout

    try:
        body = _read_bounded(
            response,
            MAX_WEB_RESPONSE_BYTES,
            deadline,
            clock,
            read_timeout_setter,
        )
    except _ReadDeadlineExceeded as exc:
        raise ValueError(f"Timed out reading response from {url}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not fetch {url}: {exc}") from exc

    if len(body) > MAX_WEB_RESPONSE_BYTES:
        raise ValueError(
            f"Response too large: over {MAX_WEB_RESPONSE_BYTES} bytes from {url}"
        )

    content_type = _content_type(response)
    if content_type and content_type not in TEXTUAL_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {content_type} from {url}. "
            f"Supported: {', '.join(TEXTUAL_CONTENT_TYPES)}"
        )

    html = body.decode("utf-8", errors="replace")
    text = extract_text(html)

    if not text:
        raise ValueError(f"Page has no readable text: {url}")

    if len(text) > MAX_WEB_TEXT_CHARS:
        raise ValueError(
            f"Page extracted text too large: {len(text)} characters "
            f"(limit {MAX_WEB_TEXT_CHARS})"
        )

    return text
