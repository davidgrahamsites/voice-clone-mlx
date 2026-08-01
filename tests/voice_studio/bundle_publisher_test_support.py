"""Shared inputs for bundle publisher tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voiceclonegpt.training.bundle_publisher import (
    BundlePublicationRequest,
    LicenseLayer,
    PayloadFile,
    Provenance,
    ReferenceClip,
    RuntimeParityReport,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"


class FakeCopier:
    def copy(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def request(tmp_path: Path, **overrides: object) -> BundlePublicationRequest:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    contents = {
        "source-release.json": b'{"source_release_id":"source-001"}\n',
        "checkpoint.bin": b"source checkpoint",
        "variant.json": b'{"runtime":"mlx_qwen"}\n',
        "model.bin": b"runtime candidate",
        "neutral.wav": b"RIFF reference",
        "usage.json": b'{"scope":"personal-noncommercial"}\n',
        "code.txt": b"code notice",
        "weights.txt": b"weights notice",
        "derivative.txt": b"derivative notice",
        "runtime.txt": b"runtime notice",
        "output.txt": b"output notice",
    }
    contents["runtime-report.json"] = _json(
        {
            "decision": "accepted",
            "mandatory_thresholds_passed": True,
            "source_release_sha256": sha(contents["source-release.json"]),
            "runtime_candidate_sha256": sha(contents["model.bin"]),
            "parity_set_sha256": "a" * 64,
            "approval": {
                "approver_name": "Local Owner",
                "approved_at": "2026-08-01T14:01:00+00:00",
            },
        }
    )
    contents["references.json"] = _json(
        {
            "default_reference_id": "neutral",
            "references": [{
                "reference_id": "neutral",
                "path": "references/neutral.wav",
                "sha256": sha(contents["neutral.wav"]),
                "transcript": "A neutral reference.",
                "style": "neutral",
                "consented": True,
            }],
        }
    )
    for name, data in contents.items():
        (inputs / name).write_bytes(data)

    paths = {
        "source-release.json": "source/source-release.json",
        "checkpoint.bin": "source/checkpoint/checkpoint.bin",
        "variant.json": "runtimes/mlx_qwen/variant.json",
        "model.bin": "runtimes/mlx_qwen/model/model.bin",
        "runtime-report.json": "evaluation/runtime-reports/mlx_qwen.json",
        "references.json": "references/references.json",
        "neutral.wav": "references/neutral.wav",
        "usage.json": "licenses/usage.json",
        "code.txt": "licenses/notices/code.txt",
        "weights.txt": "licenses/notices/weights.txt",
        "derivative.txt": "licenses/notices/derivative.txt",
        "runtime.txt": "licenses/notices/runtime.txt",
        "output.txt": "licenses/notices/output.txt",
    }
    payloads = tuple(
        PayloadFile(inputs / name, destination, sha(contents[name]))
        for name, destination in paths.items()
    )
    fields = dict(
        bundles_root=tmp_path / "models" / "versions",
        bundle_id="voice-001@0.1.0",
        voice_id="voice-001",
        model_version="0.1.0",
        bundle_revision=1,
        created_at="2026-08-01T14:00:00+00:00",
        promoted_at="2026-08-01T14:02:00+00:00",
        source_release_path="source/source-release.json",
        source_artifact_kind="fine_tuned_adapter",
        source_checkpoint_format="safetensors-adapter",
        source_checkpoint_revision="checkpoint-revision",
        runtime_variant_id="mlx_qwen",
        runtime_backend_id="qwen3-tts-mlx",
        runtime_format="mlx",
        runtime_artifact_path="runtimes/mlx_qwen/model/model.bin",
        runtime_variant_manifest_path="runtimes/mlx_qwen/variant.json",
        runtime_report_path="evaluation/runtime-reports/mlx_qwen.json",
        payloads=payloads,
        parity=RuntimeParityReport(
            decision="accepted",
            mandatory_thresholds_passed=True,
            source_release_sha256=sha(contents["source-release.json"]),
            runtime_candidate_sha256=sha(contents["model.bin"]),
            parity_set_sha256="a" * 64,
            report_sha256=sha(contents["runtime-report.json"]),
            approver_name="Local Owner",
            approved_at="2026-08-01T14:01:00+00:00",
        ),
        provenance=Provenance(
            source_release_id="source-001",
            source_release_sha256=sha(contents["source-release.json"]),
            dataset_id="dataset-001",
            dataset_sha256="b" * 64,
            training_run_id="run-001",
            training_manifest_sha256="c" * 64,
            source_evaluation_id="source-eval-001",
            source_evaluation_sha256="d" * 64,
            base_model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            base_model_revision="base-revision",
            base_model_sha256="e" * 64,
            converter_id="mlx-audio",
            converter_revision="converter-revision",
            converter_status="community",
        ),
        references=(ReferenceClip(
            reference_id="neutral",
            path="references/neutral.wav",
            sha256=sha(contents["neutral.wav"]),
            transcript="A neutral reference.",
            style="neutral",
            consented=True,
            is_default=True,
        ),),
        reference_index_path="references/references.json",
        licenses=tuple(
            LicenseLayer(kind, paths[name], sha(contents[name]))
            for kind, name in (
                ("usage", "usage.json"),
                ("code", "code.txt"),
                ("source_weights", "weights.txt"),
                ("derivative", "derivative.txt"),
                ("converter_runtime", "runtime.txt"),
                ("output_use", "output.txt"),
            )
        ),
        audio_sample_rate=24000,
    )
    fields.update(overrides)
    return BundlePublicationRequest(**fields)  # type: ignore[arg-type]
