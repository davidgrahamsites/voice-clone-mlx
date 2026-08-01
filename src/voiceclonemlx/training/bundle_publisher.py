"""Publish one approved runtime variant as an immutable Bundle Contract v1 tree.

This standard-library module owns filesystem publication only. It never loads,
registers, selects, or promotes a runtime and never writes ``promoted.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Protocol
from uuid import uuid4

class BundlePublicationError(ValueError):
    """The requested bundle cannot be safely and completely published."""

@dataclass(frozen=True)
class PayloadFile:
    source: Path; destination: str; sha256: str

@dataclass(frozen=True)
class RuntimeParityReport:
    decision: str; mandatory_thresholds_passed: bool
    source_release_sha256: str; runtime_candidate_sha256: str
    parity_set_sha256: str; report_sha256: str
    approver_name: str; approved_at: str

@dataclass(frozen=True)
class Provenance:
    source_release_id: str; source_release_sha256: str
    dataset_id: str; dataset_sha256: str
    training_run_id: str; training_manifest_sha256: str
    source_evaluation_id: str; source_evaluation_sha256: str
    base_model_id: str; base_model_revision: str; base_model_sha256: str
    converter_id: str; converter_revision: str; converter_status: str

@dataclass(frozen=True)
class ReferenceClip:
    reference_id: str; path: str; sha256: str
    transcript: str; style: str
    consented: bool; is_default: bool

@dataclass(frozen=True)
class LicenseLayer:
    kind: str; path: str; sha256: str

@dataclass(frozen=True)
class BundlePublicationRequest:
    bundles_root: Path; bundle_id: str; voice_id: str; model_version: str
    bundle_revision: int; created_at: str; promoted_at: str
    source_release_path: str; source_artifact_kind: str
    source_checkpoint_format: str; source_checkpoint_revision: str
    runtime_variant_id: str; runtime_backend_id: str; runtime_format: str
    runtime_artifact_path: str; runtime_variant_manifest_path: str
    runtime_report_path: str; payloads: tuple[PayloadFile, ...]
    parity: RuntimeParityReport; provenance: Provenance
    references: tuple[ReferenceClip, ...]; reference_index_path: str
    licenses: tuple[LicenseLayer, ...]; audio_sample_rate: int

@dataclass(frozen=True)
class PublishedBundle:
    path: Path; bundle_sha256: str

class PayloadCopier(Protocol):
    def copy(self, source: Path, destination: Path) -> None: ...

class _FileCopier:
    def copy(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination, follow_symlinks=False)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"

def _payload_map(request: BundlePublicationRequest) -> dict[str, PayloadFile]:
    return {item.destination: item for item in request.payloads}

def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def _validate_payloads(request: BundlePublicationRequest) -> None:
    if not request.payloads:
        raise BundlePublicationError("At least one payload is required.")
    destinations = [item.destination for item in request.payloads]
    if len(destinations) != len(set(destinations)):
        raise BundlePublicationError("Payload destinations must be unique.")
    for payload in request.payloads:
        relative = Path(payload.destination)
        if (
            not isinstance(payload.destination, str)
            or not payload.destination
            or "\\" in payload.destination
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != payload.destination
            or payload.destination in {"bundle.json", "checksums.sha256"}
        ):
            raise BundlePublicationError("Every payload needs a safe relative destination.")
        source = Path(payload.source)
        if source.is_symlink():
            raise BundlePublicationError("Payload sources may not be symlinks.")
        if not source.is_file():
            raise BundlePublicationError(f"Payload source is not a file: {source}")
        if not _is_sha256(payload.sha256) or _sha256(source) != payload.sha256:
            raise BundlePublicationError(
                f"Payload checksum does not match source bytes: {payload.destination}"
            )

def _validate_parity(request: BundlePublicationRequest) -> None:
    parity = request.parity
    if parity.decision != "accepted":
        raise BundlePublicationError("Runtime parity must be explicitly accepted.")
    if parity.mandatory_thresholds_passed is not True:
        raise BundlePublicationError("Every mandatory parity threshold must pass.")
    if not isinstance(parity.approver_name, str) or not parity.approver_name.strip():
        raise BundlePublicationError("A named parity approver is required.")
    payloads = _payload_map(request)
    bindings = (
        ("source", request.source_release_path, parity.source_release_sha256),
        ("runtime", request.runtime_artifact_path, parity.runtime_candidate_sha256),
        ("report", request.runtime_report_path, parity.report_sha256),
    )
    for label, path, approved_checksum in bindings:
        payload = payloads.get(path)
        if payload is None or payload.sha256 != approved_checksum:
            raise BundlePublicationError(
                f"The accepted parity report does not bind the exact {label} payload."
            )
    if request.provenance.source_release_sha256 != parity.source_release_sha256:
        raise BundlePublicationError(
            "The accepted parity report does not bind the exact source provenance."
        )

def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())

def _is_utc(value: object) -> bool:
    if not _is_text(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)

def _validate_evidence(request: BundlePublicationRequest) -> None:
    provenance = request.provenance
    text_fields = (
        provenance.source_release_id,
        provenance.dataset_id,
        provenance.training_run_id,
        provenance.source_evaluation_id,
        provenance.base_model_id,
        provenance.base_model_revision,
        provenance.converter_id,
        provenance.converter_revision,
    )
    checksum_fields = (
        provenance.source_release_sha256,
        provenance.dataset_sha256,
        provenance.training_manifest_sha256,
        provenance.source_evaluation_sha256,
        provenance.base_model_sha256,
        request.parity.parity_set_sha256,
        request.parity.report_sha256,
    )
    if (
        not all(_is_text(value) for value in text_fields)
        or not all(_is_sha256(value) for value in checksum_fields)
        or provenance.converter_status not in {"official", "community"}
    ):
        raise BundlePublicationError("Complete exact provenance is required.")
    if not _is_utc(request.parity.approved_at):
        raise BundlePublicationError("Parity provenance needs a UTC approval time.")

    payloads = _payload_map(request)
    required_license_kinds = {
        "usage", "code", "source_weights", "derivative",
        "converter_runtime", "output_use",
    }
    license_kinds = [item.kind for item in request.licenses]
    if set(license_kinds) != required_license_kinds or len(license_kinds) != len(
        set(license_kinds)
    ):
        raise BundlePublicationError("Every required license layer is required exactly once.")
    for layer in request.licenses:
        payload = payloads.get(layer.path)
        if payload is None or payload.sha256 != layer.sha256:
            raise BundlePublicationError("Every license layer must bind an included payload.")

    if request.reference_index_path not in payloads:
        raise BundlePublicationError("The reference index must be included by value.")
    reference_ids = [item.reference_id for item in request.references]
    defaults = [item for item in request.references if item.is_default]
    if (
        not request.references
        or len(reference_ids) != len(set(reference_ids))
        or len(defaults) != 1
        or any(
            not all(_is_text(value) for value in (
                item.reference_id, item.path, item.transcript, item.style
            ))
            or item.consented is not True
            or item.path not in payloads
            or payloads[item.path].sha256 != item.sha256
            for item in request.references
        )
    ):
        raise BundlePublicationError(
            "References require unique identities, exact payloads, consent, and one default reference."
        )

    try:
        index = json.loads(payloads[request.reference_index_path].source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundlePublicationError("The reference index is not valid JSON.") from exc
    expected_references = [{
        "reference_id": item.reference_id,
        "path": item.path,
        "sha256": item.sha256,
        "transcript": item.transcript,
        "style": item.style,
        "consented": True,
    } for item in request.references]
    if (
        not isinstance(index, dict)
        or index.get("default_reference_id") != defaults[0].reference_id
        or index.get("references") != expected_references
    ):
        raise BundlePublicationError(
            "The reference index bytes do not match the consented reference records."
        )

def _validate_contract_shape(request: BundlePublicationRequest) -> None:
    if request.bundle_id != f"{request.voice_id}@{request.model_version}":
        raise BundlePublicationError("bundle_id must exactly match voice_id@model_version.")
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", request.model_version):
        raise BundlePublicationError("model_version must be semantic MAJOR.MINOR.PATCH.")
    if not isinstance(request.bundle_revision, int) or isinstance(request.bundle_revision, bool) or request.bundle_revision < 1:
        raise BundlePublicationError("bundle_revision must be a positive integer.")
    if not _is_utc(request.created_at) or not _is_utc(request.promoted_at):
        raise BundlePublicationError("Bundle timestamps must be UTC.")
    if request.source_artifact_kind not in {"fine_tuned_full", "fine_tuned_adapter"}:
        raise BundlePublicationError("The source artifact must be a learned model.")
    if not all(_is_text(value) for value in (
        request.voice_id,
        request.source_checkpoint_format,
        request.source_checkpoint_revision,
        request.runtime_variant_id,
        request.runtime_backend_id,
        request.runtime_format,
    )):
        raise BundlePublicationError("Required bundle identities and formats must be present.")
    if not isinstance(request.audio_sample_rate, int) or isinstance(request.audio_sample_rate, bool) or request.audio_sample_rate <= 0:
        raise BundlePublicationError("The audio sample rate must be a positive integer.")

    payloads = _payload_map(request)
    exact_paths = (
        request.source_release_path,
        request.runtime_variant_manifest_path,
        request.runtime_artifact_path,
        request.runtime_report_path,
        request.reference_index_path,
    )
    if any(path not in payloads for path in exact_paths) or not any(
        path.startswith("source/checkpoint/") for path in payloads
    ):
        raise BundlePublicationError("Every required Bundle Contract v1 payload is required.")

    report_payload = payloads[request.runtime_report_path]
    try:
        report = json.loads(report_payload.source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundlePublicationError("The runtime parity report bytes are not valid JSON.") from exc
    expected = {
        "decision": request.parity.decision,
        "mandatory_thresholds_passed": request.parity.mandatory_thresholds_passed,
        "source_release_sha256": request.parity.source_release_sha256,
        "runtime_candidate_sha256": request.parity.runtime_candidate_sha256,
        "parity_set_sha256": request.parity.parity_set_sha256,
    }
    if not isinstance(report, dict) or any(report.get(key) != value for key, value in expected.items()):
        raise BundlePublicationError("The runtime parity report bytes do not match accepted bindings.")
    approval = report.get("approval")
    if not isinstance(approval, dict) or (
        approval.get("approver_name") != request.parity.approver_name
        or approval.get("approved_at") != request.parity.approved_at
    ):
        raise BundlePublicationError("The runtime parity report bytes do not match accepted approval.")

def publish_bundle(
    request: BundlePublicationRequest,
    copier: PayloadCopier | None = None,
) -> PublishedBundle:
    """Copy, verify, and atomically expose one complete immutable bundle."""
    _validate_payloads(request)
    _validate_parity(request)
    _validate_evidence(request)
    _validate_contract_shape(request)
    copier = copier or _FileCopier()
    root = Path(request.bundles_root)
    destination = root / request.bundle_id
    stage = root / f".{request.bundle_id}.{uuid4().hex}.partial"
    if destination.exists() or destination.is_symlink():
        raise BundlePublicationError("The destination bundle already exists; refusing overwrite.")

    root.mkdir(parents=True, exist_ok=True)
    try:
        stage.mkdir()
        for payload in request.payloads:
            target = stage / payload.destination
            target.parent.mkdir(parents=True, exist_ok=True)
            copier.copy(Path(payload.source), target)
            if target.is_symlink() or not target.is_file() or _sha256(target) != payload.sha256:
                raise BundlePublicationError(
                    f"Copied payload checksum mismatch: {payload.destination}"
                )

        provenance, parity = request.provenance, request.parity
        license_paths = {item.kind: item.path for item in request.licenses}
        manifest = {
            "bundle_schema_version": "1.0.0", "bundle_id": request.bundle_id,
            "voice_id": request.voice_id, "model_version": request.model_version,
            "bundle_revision": request.bundle_revision, "created_at": request.created_at,
            "promoted_at": request.promoted_at,
            "source_model": {
                "artifact_kind": request.source_artifact_kind, "checkpoint_path": "source/checkpoint",
                "source_release_id": provenance.source_release_id,
                "source_release_path": request.source_release_path, "sha256": provenance.source_release_sha256,
                "checkpoint_format": request.source_checkpoint_format, "checkpoint_revision": request.source_checkpoint_revision,
                "base_model_id": provenance.base_model_id, "base_model_revision": provenance.base_model_revision,
                "base_model_sha256": provenance.base_model_sha256,
                "license_references": [
                    license_paths[kind] for kind in ("code", "source_weights", "derivative")
                ],
            },
            "runtime_variants": [{
                "id": request.runtime_variant_id, "backend": request.runtime_backend_id,
                "format": request.runtime_format, "artifact": request.runtime_artifact_path,
                "sha256": parity.runtime_candidate_sha256,
                "variant_manifest": request.runtime_variant_manifest_path, "parity_report": request.runtime_report_path,
                "converter_id": provenance.converter_id, "converter_revision": provenance.converter_revision,
                "upstream_status": provenance.converter_status,
                "license_references": [license_paths["converter_runtime"], license_paths["output_use"]],
            }],
            "references": {
                "index": request.reference_index_path, "default_reference_id": next(
                    item.reference_id for item in request.references if item.is_default),
                "styles": sorted({item.style for item in request.references}),
            },
            "training_provenance": {
                "dataset_id": provenance.dataset_id, "dataset_sha256": provenance.dataset_sha256,
                "training_run_id": provenance.training_run_id,
                "training_manifest_sha256": provenance.training_manifest_sha256,
                "source_evaluation_id": provenance.source_evaluation_id,
                "source_evaluation_sha256": provenance.source_evaluation_sha256,
                "parity_set_sha256": parity.parity_set_sha256,
            },
            "compatibility": {"minimum_loader_contract_version": "1.0.0"},
            "audio_contract": {"sample_rate": request.audio_sample_rate, "channels": 1, "encoding": "pcm_s16le"},
            "license": {"usage": license_paths["usage"], "layers": license_paths},
            "integrity": {"algorithm": "sha256", "checksum_manifest": "checksums.sha256"},
        }
        (stage / "bundle.json").write_bytes(_json_bytes(manifest))
        files = sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        checksum_text = "".join(f"{_sha256(stage / name)}  {name}\n" for name in files)
        (stage / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        bundle_sha256 = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
        if destination.exists() or destination.is_symlink():
            raise BundlePublicationError("The destination bundle already exists; refusing overwrite.")
        os.rename(stage, destination)
        return PublishedBundle(destination, bundle_sha256)
    except Exception as exc:
        if stage.exists():
            shutil.rmtree(stage)
        if isinstance(exc, BundlePublicationError):
            raise
        raise BundlePublicationError(f"Bundle publication failed: {exc}") from exc
