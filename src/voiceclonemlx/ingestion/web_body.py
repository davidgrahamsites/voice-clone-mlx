"""Read a response body under a byte cap and a wall-clock deadline.

Transport-agnostic: anything exposing `.read(size)` works. Knows nothing about
URLs, HTML, or fetch policy.
"""

READ_CHUNK_BYTES = 64 * 1024


def _apply_read_timeout(response, seconds: float) -> None:
    """Set the remaining deadline as the socket timeout for the next read.

    Without this, the overall deadline is only checked *between* reads, so one
    blocking read could outlast it. Responses that expose no socket (test
    fakes, exotic transports) are left alone.

    Args:
        response: The response object
        seconds: Remaining time before the deadline
    """
    candidates = (
        getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None),
        getattr(getattr(response, "fp", None), "_sock", None),
        getattr(response, "_sock", None),
    )

    for sock in candidates:
        setter = getattr(sock, "settimeout", None)
        if callable(setter):
            setter(seconds)
            return


class _ReadDeadlineExceeded(Exception):
    """Internal: the body read outlasted its deadline.

    A private type so the deadline is never confused with a ValueError raised
    by the transport, which must be normalized instead.
    """


def _read_bounded(
    response, limit: int, deadline: float, clock, timeout_setter=None
) -> bytes:
    """Read a response body under both a byte cap and a wall-clock deadline.

    Args:
        response: Object with `.read(size)`
        limit: Maximum bytes to accept
        deadline: Clock value past which the read is abandoned
        clock: Callable returning the current time
        timeout_setter: Callable `(response, seconds)` applying the remaining
            deadline to the underlying socket. Defaults to
            `_apply_read_timeout`; injectable for tests.

    Returns:
        Up to `limit + 1` bytes — one past the cap, so the caller can detect
        an oversized body without ever holding it

    Raises:
        _ReadDeadlineExceeded: If the deadline passes before the body is read
    """
    if timeout_setter is None:
        timeout_setter = _apply_read_timeout

    chunks = []
    total = 0

    while total <= limit:
        remaining = deadline - clock()
        if remaining <= 0:
            raise _ReadDeadlineExceeded(
                f"Timed out reading response after {total} bytes"
            )

        # Bound the individual blocking read too, so one slow read cannot
        # outlast the overall deadline while we wait for it to return.
        timeout_setter(response, remaining)

        chunk = response.read(min(READ_CHUNK_BYTES, limit + 1 - total))
        if not chunk:
            break

        chunks.append(chunk)
        total += len(chunk)

    return b"".join(chunks)
