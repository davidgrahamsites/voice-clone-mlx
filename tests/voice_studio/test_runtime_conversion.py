"""Public contract tests for one provider-neutral MLX conversion attempt."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from voiceclonegpt.training.runtime_conversion import (
    ConvertedPayload,
    InvalidConversionRequest,
    InvalidConversionResult,
    SourceModelRelease,
    ConversionRequest,
    coordinate_mlx_conversion,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def accepted_request() -> ConversionRequest:
    return ConversionRequest(
        conversion_id="conversion-001",
        source_release=SourceModelRelease(
            source_release_id="source-001",
            approval_state="accepted",
            checkpoint_path="source/checkpoint",
            checkpoint_sha256=SHA_A,
            source_release_sha256=SHA_A,
            provider_id="qwen3-tts",
            provider_revision="revision-123",
        ),
        converter_id="mlx-audio",
        converter_revision="mlx-audio-1.2.3",
        converter_status="community",
        target_format="mlx",
        runtime_version="0.24.0",
        quantization="none",
        tensor_dtype_mapping="float32->float16",
        parity_set_sha256=SHA_B,
    )


class SuccessfulConverter:
    def __init__(self) -> None:
        self.requests = []

    def convert(self, request: ConversionRequest) -> ConvertedPayload:
        self.requests.append(request)
        return ConvertedPayload(
            candidate_id="mlx-candidate-001",
            artifact_path="candidate/model",
            artifact_sha256=SHA_C,
            source_release_sha256=SHA_A,
            converter_id=request.converter_id,
            converter_revision=request.converter_revision,
            target_format=request.target_format,
            runtime_version=request.runtime_version,
            quantization=request.quantization,
            tensor_dtype_mapping=request.tensor_dtype_mapping,
        )


def test_conversion_binds_approved_source_to_pending_mlx_candidate():
    converter = SuccessfulConverter()

    manifest = coordinate_mlx_conversion(accepted_request(), converter)

    assert converter.requests == [accepted_request()]
    assert manifest.status == "converted"
    assert manifest.source_release_id == "source-001"
    assert manifest.source_release_sha256 == SHA_A
    assert manifest.candidate_id == "mlx-candidate-001"
    assert manifest.candidate_sha256 == SHA_C
    assert manifest.parity_set_sha256 == SHA_B
    assert manifest.parity_status == "pending"


def test_unknown_converter_provenance_status_is_refused_before_converter_call():
    converter = SuccessfulConverter()
    request = replace(accepted_request(), converter_status="unverified")

    with pytest.raises(InvalidConversionRequest, match="status"):
        coordinate_mlx_conversion(request, converter)

    assert converter.requests == []


@pytest.mark.parametrize(
    "request_factory",
    (
        lambda: replace(
            accepted_request(),
            source_release=replace(accepted_request().source_release, approval_state="pending"),
        ),
        lambda: replace(
            accepted_request(),
            source_release=replace(accepted_request().source_release, checkpoint_sha256=SHA_C),
        ),
        lambda: replace(accepted_request(), parity_set_sha256="not-a-checksum"),
    ),
)
def test_invalid_source_or_parity_evidence_never_calls_converter(request_factory):
    converter = SuccessfulConverter()

    with pytest.raises(InvalidConversionRequest):
        coordinate_mlx_conversion(request_factory(), converter)

    assert converter.requests == []


@pytest.mark.parametrize(
    "field, value",
    (
        ("source_release_sha256", SHA_B),
        ("converter_revision", "wrong-revision"),
        ("target_format", "pytorch"),
        ("runtime_version", "wrong-runtime"),
        ("quantization", "int4"),
        ("tensor_dtype_mapping", "float16->int4"),
    ),
)
def test_converter_payload_must_preserve_requested_provenance_and_mapping(field, value):
    class MismatchedConverter(SuccessfulConverter):
        def convert(self, request):
            payload = super().convert(request)
            return replace(payload, **{field: value})

    with pytest.raises(InvalidConversionResult, match="payload"):
        coordinate_mlx_conversion(accepted_request(), MismatchedConverter())


def test_converter_failure_is_not_retried():
    class FailingConverter:
        calls = 0

        def convert(self, request):
            self.calls += 1
            raise RuntimeError("converter stopped")

    converter = FailingConverter()
    with pytest.raises(RuntimeError, match="converter stopped"):
        coordinate_mlx_conversion(accepted_request(), converter)

    assert converter.calls == 1


def test_module_never_imports_or_registers_a_runtime():
    import voiceclonegpt.training.runtime_conversion as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name if isinstance(node, ast.Import) else node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }

    assert imports <= {"__future__", "dataclasses", "typing"}
    assert "runtime_registry" not in Path(module.__file__).read_text(encoding="utf-8")
