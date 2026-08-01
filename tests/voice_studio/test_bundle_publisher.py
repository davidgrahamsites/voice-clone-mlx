"""Bundle Contract v1 publication validation."""

import json
from pathlib import Path
from dataclasses import replace

import pytest

from bundle_publisher_test_support import FakeCopier, request, sha
from voiceclonegpt.training.bundle_publisher import (
    BundlePublicationError,
    ReferenceClip,
    publish_bundle,
)


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
