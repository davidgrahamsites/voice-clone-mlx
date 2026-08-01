"""Bundle Contract v1 publication through one filesystem seam."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
from dataclasses import replace

import pytest

from voiceclonegpt.training.bundle_publisher import (
    BundlePublicationRequest,
    BundlePublicationError,
    LicenseLayer,
    PayloadFile,
    Provenance,
    ReferenceClip,
    RuntimeParityReport,
    publish_bundle,
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


def test_accepted_bound_evidence_publishes_one_complete_bundle(tmp_path: Path):
    published = publish_bundle(request(tmp_path), FakeCopier())

    assert published.path == tmp_path / "models" / "versions" / "voice-001@0.1.0"
    assert published.path.is_dir()
    manifest = json.loads((published.path / "bundle.json").read_text())
    assert manifest["bundle_schema_version"] == "1.0.0"
    assert manifest["bundle_id"] == "voice-001@0.1.0"
    assert manifest["runtime_variants"][0]["id"] == "mlx_qwen"
    assert not list(published.path.parent.glob("*.partial"))


@pytest.mark.parametrize(
    ("parity_change", "message"),
    (
        ({"decision": "pending_human_review"}, "accepted"),
        ({"mandatory_thresholds_passed": False}, "threshold"),
        ({"source_release_sha256": "f" * 64}, "source"),
        ({"runtime_candidate_sha256": "f" * 64}, "runtime"),
        ({"report_sha256": "f" * 64}, "report"),
        ({"approver_name": " "}, "approver"),
    ),
)
def test_unapproved_or_unbound_parity_leaves_no_bundle(
    tmp_path: Path, parity_change: dict[str, object], message: str
):
    publication = request(tmp_path)
    publication = replace(
        publication,
        parity=replace(publication.parity, **parity_change),
    )

    with pytest.raises(BundlePublicationError, match=message):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


def test_changed_payload_bytes_are_refused_before_staging(tmp_path: Path):
    publication = request(tmp_path)
    publication.payloads[-1].source.write_bytes(b"tampered")

    with pytest.raises(BundlePublicationError, match="checksum"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


@pytest.mark.parametrize("destination", ("../escape", "/absolute/payload"))
def test_unsafe_payload_destination_is_refused(tmp_path: Path, destination: str):
    publication = request(tmp_path)
    changed = replace(publication.payloads[-1], destination=destination)
    publication = replace(publication, payloads=publication.payloads[:-1] + (changed,))

    with pytest.raises(BundlePublicationError, match="safe relative"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


def test_symlink_payload_is_refused_even_when_it_points_to_a_file(tmp_path: Path):
    publication = request(tmp_path)
    original = publication.payloads[-1]
    link = original.source.with_name("linked-output.txt")
    link.symlink_to(original.source)
    publication = replace(
        publication,
        payloads=publication.payloads[:-1] + (replace(original, source=link),),
    )

    with pytest.raises(BundlePublicationError, match="symlink"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_id", " "),
        ("training_manifest_sha256", "bad"),
        ("base_model_revision", ""),
        ("converter_status", "unverified"),
    ),
)
def test_incomplete_provenance_is_refused_before_staging(
    tmp_path: Path, field: str, value: object
):
    publication = request(tmp_path)
    publication = replace(
        publication,
        provenance=replace(publication.provenance, **{field: value}),
    )

    with pytest.raises(BundlePublicationError, match="provenance"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


def test_every_license_layer_is_required_and_bound_to_payload(tmp_path: Path):
    publication = request(tmp_path)
    publication = replace(
        publication,
        licenses=tuple(
            layer for layer in publication.licenses if layer.kind != "output_use"
        ),
    )

    with pytest.raises(BundlePublicationError, match="license"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


@pytest.mark.parametrize(
    "references",
    (
        (),
        (ReferenceClip("neutral", "references/neutral.wav", "0" * 64,
                       "A neutral reference.", "neutral", False, True),),
        (ReferenceClip("neutral", "references/neutral.wav", "0" * 64,
                       "A neutral reference.", "neutral", True, False),),
    ),
)
def test_reference_library_requires_one_consented_default(
    tmp_path: Path, references: tuple[ReferenceClip, ...]
):
    publication = request(tmp_path)
    if references:
        references = tuple(
            replace(item, sha256=publication.references[0].sha256)
            for item in references
        )
    publication = replace(publication, references=references)

    with pytest.raises(BundlePublicationError, match="reference"):
        publish_bundle(publication, FakeCopier())

    assert not publication.bundles_root.exists()


def test_checksum_manifest_is_sorted_complete_and_verified(tmp_path: Path):
    published = publish_bundle(request(tmp_path), FakeCopier())
    checksum_file = published.path / "checksums.sha256"
    rows = checksum_file.read_text().splitlines()
    names = [row.split("  ", 1)[1] for row in rows]
    expected = sorted(
        path.relative_to(published.path).as_posix()
        for path in published.path.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )

    assert names == expected
    for row in rows:
        digest, name = row.split("  ", 1)
        assert digest == sha((published.path / name).read_bytes())
    assert published.bundle_sha256 == sha(checksum_file.read_bytes())


def test_existing_destination_is_never_overwritten(tmp_path: Path):
    publication = request(tmp_path)
    destination = publication.bundles_root / publication.bundle_id
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("existing")

    with pytest.raises(BundlePublicationError, match="overwrite"):
        publish_bundle(publication, FakeCopier())

    assert marker.read_text() == "existing"


@pytest.mark.parametrize("failure", ("raise", "corrupt"))
def test_copy_failure_or_corruption_leaves_no_partial_bundle(
    tmp_path: Path, failure: str
):
    publication = request(tmp_path)

    class BrokenCopier:
        def copy(self, source: Path, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if failure == "raise":
                raise OSError("injected copy failure")
            destination.write_bytes(b"wrong bytes")

    with pytest.raises(BundlePublicationError):
        publish_bundle(publication, BrokenCopier())

    assert not (publication.bundles_root / publication.bundle_id).exists()
    assert not list(publication.bundles_root.glob("*.partial"))


def test_atomic_rename_failure_leaves_no_partial_bundle(tmp_path: Path, monkeypatch):
    import voiceclonegpt.training.bundle_publisher as module

    publication = request(tmp_path)

    def fail_rename(source: Path, destination: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(module.os, "rename", fail_rename)

    with pytest.raises(BundlePublicationError, match="rename failure"):
        publish_bundle(publication, FakeCopier())

    assert not (publication.bundles_root / publication.bundle_id).exists()
    assert not list(publication.bundles_root.glob("*.partial"))


def test_published_manifest_is_accepted_by_shared_bundle_reader(tmp_path: Path):
    from voiceclonegpt.shared.bundle_reader import read_bundle

    published = publish_bundle(request(tmp_path), FakeCopier())

    assert read_bundle(published.path).bundle_id == "voice-001@0.1.0"


def test_module_is_detachable_and_has_no_runtime_or_catalog_dependency():
    import voiceclonegpt.training.bundle_publisher as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert imported <= {
        "__future__", "dataclasses", "datetime", "hashlib", "json", "os",
        "pathlib", "re", "shutil", "typing", "uuid",
    }
    assert "runtime_registry" not in source
    assert "voice_model_catalog" not in source
    assert not any(
        isinstance(node, ast.Constant) and node.value == "promoted.json"
        for node in ast.walk(ast.parse(source))
    )


def test_runtime_report_bytes_must_match_the_accepted_bindings(tmp_path: Path):
    publication = request(tmp_path)
    report = next(
        item for item in publication.payloads
        if item.destination == publication.runtime_report_path
    )
    changed_bytes = _json({
        "decision": "accepted",
        "mandatory_thresholds_passed": True,
        "source_release_sha256": "f" * 64,
        "runtime_candidate_sha256": publication.parity.runtime_candidate_sha256,
        "parity_set_sha256": publication.parity.parity_set_sha256,
        "approval": {
            "approver_name": publication.parity.approver_name,
            "approved_at": publication.parity.approved_at,
        },
    })
    report.source.write_bytes(changed_bytes)
    changed_report = replace(report, sha256=sha(changed_bytes))
    publication = replace(
        publication,
        payloads=tuple(
            changed_report if item is report else item for item in publication.payloads
        ),
        parity=replace(publication.parity, report_sha256=sha(changed_bytes)),
    )

    with pytest.raises(BundlePublicationError, match="report bytes"):
        publish_bundle(publication, FakeCopier())


def test_required_bundle_payload_cannot_be_omitted(tmp_path: Path):
    publication = request(tmp_path)
    publication = replace(
        publication,
        payloads=tuple(
            item for item in publication.payloads
            if not item.destination.startswith("source/checkpoint/")
        ),
    )

    with pytest.raises(BundlePublicationError, match="required"):
        publish_bundle(publication, FakeCopier())


def test_bundle_identity_must_match_voice_and_model_version(tmp_path: Path):
    publication = request(tmp_path, bundle_id="../wrong")

    with pytest.raises(BundlePublicationError, match="bundle_id"):
        publish_bundle(publication, FakeCopier())


def test_reference_index_bytes_must_match_consented_records(tmp_path: Path):
    publication = request(tmp_path)
    index = next(
        item for item in publication.payloads
        if item.destination == publication.reference_index_path
    )
    changed_bytes = _json({"default_reference_id": "someone-else", "references": []})
    index.source.write_bytes(changed_bytes)
    changed_index = replace(index, sha256=sha(changed_bytes))
    publication = replace(
        publication,
        payloads=tuple(
            changed_index if item is index else item for item in publication.payloads
        ),
    )

    with pytest.raises(BundlePublicationError, match="reference index"):
        publish_bundle(publication, FakeCopier())
