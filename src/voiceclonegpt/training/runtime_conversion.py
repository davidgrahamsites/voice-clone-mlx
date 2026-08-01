"""Coordinate one approved source-to-MLX conversion without performing it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


_LOWER_HEX = frozenset("0123456789abcdef")
_CONVERTER_STATUSES = frozenset(("official", "community"))


class InvalidConversionRequest(ValueError):
    """The source evidence cannot cross the converter boundary."""


class InvalidConversionResult(ValueError):
    """The converter did not return a bound, checksummed MLX payload."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidConversionRequest(f"{field} must be non-blank text.")


@dataclass(frozen=True)
class SourceModelRelease:
    """The approved immutable source checkpoint being converted."""

    source_release_id: str
    approval_state: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_release_sha256: str
    provider_id: str
    provider_revision: str


@dataclass(frozen=True)
class ConversionRequest:
    """All provenance and parity evidence required for one conversion attempt."""

    conversion_id: str
    source_release: SourceModelRelease
    converter_id: str
    converter_revision: str
    converter_status: str
    target_format: str
    runtime_version: str
    quantization: str
    tensor_dtype_mapping: str
    parity_set_sha256: str


@dataclass(frozen=True)
class ConvertedPayload:
    """Checksummed MLX payload reported by the injected converter."""

    candidate_id: str
    artifact_path: str
    artifact_sha256: str
    source_release_sha256: str
    converter_id: str
    converter_revision: str
    target_format: str
    runtime_version: str
    quantization: str
    tensor_dtype_mapping: str


@dataclass(frozen=True)
class ConversionManifest:
    """A pending runtime candidate; parity evaluation remains a separate gate."""

    conversion_id: str
    source_release_id: str
    source_release_sha256: str
    candidate_id: str
    candidate_path: str
    candidate_sha256: str
    converter_id: str
    converter_revision: str
    converter_status: str
    target_format: str
    runtime_version: str
    quantization: str
    tensor_dtype_mapping: str
    parity_set_sha256: str
    parity_status: str
    status: str


class RuntimeVariantConverter(Protocol):
    """Leaf boundary that performs conversion and payload checksumming."""

    def convert(self, request: ConversionRequest) -> ConvertedPayload: ...


def _validate_request(request: ConversionRequest) -> None:
    source = request.source_release
    for field in (
        "conversion_id", "converter_id", "converter_revision", "converter_status",
        "target_format", "runtime_version", "quantization", "tensor_dtype_mapping",
    ):
        _require_text(getattr(request, field), field)
    for field in (
        "source_release_id", "checkpoint_path", "provider_id", "provider_revision",
    ):
        _require_text(getattr(source, field), field)
    if source.approval_state != "accepted":
        raise InvalidConversionRequest("The source release must be accepted before conversion.")
    if request.converter_status not in _CONVERTER_STATUSES:
        raise InvalidConversionRequest(
            "converter_status must be official or community."
        )
    if not all(_is_sha256(value) for value in (
        source.checkpoint_sha256, source.source_release_sha256, request.parity_set_sha256,
    )):
        raise InvalidConversionRequest("Every source and parity checksum must be lowercase SHA-256.")
    if source.checkpoint_sha256 != source.source_release_sha256:
        raise InvalidConversionRequest("The source checkpoint checksum must match its accepted release.")


def _validate_payload(request: ConversionRequest, payload: object) -> ConvertedPayload:
    if not isinstance(payload, ConvertedPayload):
        raise InvalidConversionResult("The converter must return a ConvertedPayload.")
    for field in ("candidate_id", "artifact_path"):
        value = getattr(payload, field)
        if not isinstance(value, str) or not value.strip():
            raise InvalidConversionResult(f"The payload {field} is missing.")
    if not _is_sha256(payload.artifact_sha256):
        raise InvalidConversionResult("The payload checksum is not valid SHA-256.")
    expected = request.source_release.source_release_sha256
    if payload.source_release_sha256 != expected:
        raise InvalidConversionResult("The payload source-release checksum does not match.")
    for field in (
        "converter_id", "converter_revision", "target_format", "runtime_version",
        "quantization", "tensor_dtype_mapping",
    ):
        if getattr(payload, field) != getattr(request, field):
            raise InvalidConversionResult(f"The payload {field} does not match the request.")
    return payload


def coordinate_mlx_conversion(
    request: ConversionRequest,
    converter: RuntimeVariantConverter,
) -> ConversionManifest:
    """Make one conversion attempt; successful candidates always await parity."""
    _validate_request(request)
    payload = _validate_payload(request, converter.convert(request))
    return ConversionManifest(
        conversion_id=request.conversion_id,
        source_release_id=request.source_release.source_release_id,
        source_release_sha256=request.source_release.source_release_sha256,
        candidate_id=payload.candidate_id,
        candidate_path=payload.artifact_path,
        candidate_sha256=payload.artifact_sha256,
        converter_id=request.converter_id,
        converter_revision=request.converter_revision,
        converter_status=request.converter_status,
        target_format=request.target_format,
        runtime_version=request.runtime_version,
        quantization=request.quantization,
        tensor_dtype_mapping=request.tensor_dtype_mapping,
        parity_set_sha256=request.parity_set_sha256,
        parity_status="pending",
        status="converted",
    )
