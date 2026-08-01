"""Bundle publisher bound-artifact tests."""

from dataclasses import replace

import pytest

from bundle_publisher_test_support import FakeCopier, _json, request, sha
from voiceclonegpt.training.bundle_publisher import BundlePublicationError, publish_bundle


def test_runtime_report_bytes_must_match_the_accepted_bindings(tmp_path):
    publication = request(tmp_path)
    report = next(item for item in publication.payloads if item.destination == publication.runtime_report_path)
    changed_bytes = _json({"decision": "accepted", "mandatory_thresholds_passed": True, "source_release_sha256": "f" * 64, "runtime_candidate_sha256": publication.parity.runtime_candidate_sha256, "parity_set_sha256": publication.parity.parity_set_sha256, "approval": {"approver_name": publication.parity.approver_name, "approved_at": publication.parity.approved_at}})
    report.source.write_bytes(changed_bytes)
    changed_report = replace(report, sha256=sha(changed_bytes))
    publication = replace(publication, payloads=tuple(changed_report if item is report else item for item in publication.payloads), parity=replace(publication.parity, report_sha256=sha(changed_bytes)))

    with pytest.raises(BundlePublicationError, match="report bytes"):
        publish_bundle(publication, FakeCopier())


def test_required_bundle_payload_cannot_be_omitted(tmp_path):
    publication = request(tmp_path)
    publication = replace(publication, payloads=tuple(item for item in publication.payloads if not item.destination.startswith("source/checkpoint/")))

    with pytest.raises(BundlePublicationError, match="required"):
        publish_bundle(publication, FakeCopier())


def test_bundle_identity_must_match_voice_and_model_version(tmp_path):
    publication = request(tmp_path, bundle_id="../wrong")

    with pytest.raises(BundlePublicationError, match="bundle_id"):
        publish_bundle(publication, FakeCopier())


def test_reference_index_bytes_must_match_consented_records(tmp_path):
    publication = request(tmp_path)
    index = next(item for item in publication.payloads if item.destination == publication.reference_index_path)
    changed_bytes = _json({"default_reference_id": "someone-else", "references": []})
    index.source.write_bytes(changed_bytes)
    changed_index = replace(index, sha256=sha(changed_bytes))
    publication = replace(publication, payloads=tuple(changed_index if item is index else item for item in publication.payloads))

    with pytest.raises(BundlePublicationError, match="reference index"):
        publish_bundle(publication, FakeCopier())
