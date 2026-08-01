import hashlib
import json
from pathlib import Path

import pytest

from voiceclonemlx.shared.bundle_reader import BundleReadError, read_bundle
from voiceclonemlx.shared.model_bundle import ArtifactKind
from voiceclonemlx.shared.voice_model_catalog import VoiceModelCatalog


def _write_bundle(root: Path, *, bundle_id: str = "alex@0.1.0") -> Path:
    voice_id = bundle_id.split("@", 1)[0]
    bundle = root / bundle_id.replace("@", "_")
    payload = bundle / "runtimes" / "mlx" / "model.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"fixture-model")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": "1.0.0",
                "bundle_id": bundle_id,
                "voice_id": voice_id,
                "model_version": "0.1.0",
                "source_model": {
                    "artifact_kind": "fine_tuned_adapter",
                    "provider": "qwen3-tts",
                },
                "runtime_variants": [
                    {
                        "id": "mlx",
                        "backend": "qwen3-mlx",
                        "artifact": "runtimes/mlx/model.bin",
                        "sha256": checksum,
                    }
                ],
                "integrity": {
                    "algorithm": "sha256",
                    "checksum_manifest": "checksums.sha256",
                },
            }
        )
    )
    entries = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        relative = path.relative_to(bundle).as_posix()
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (bundle / "checksums.sha256").write_text("".join(entries), encoding="utf-8")
    return bundle


def test_bundle_reader_accepts_trained_artifact_kinds(tmp_path: Path):
    bundle = _write_bundle(tmp_path)

    verified = read_bundle(bundle)

    assert verified.bundle_id == "alex@0.1.0"
    assert verified.artifact_kind is ArtifactKind.FINE_TUNED_ADAPTER


def test_bundle_reader_rejects_payload_checksum_mismatch(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    (bundle / "runtimes" / "mlx" / "model.bin").write_bytes(b"tampered")

    with pytest.raises(BundleReadError, match="checksum"):
        read_bundle(bundle)


def test_bundle_reader_rejects_missing_payload_checksum_manifest(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    (bundle / "checksums.sha256").unlink()

    with pytest.raises(BundleReadError, match="checksum manifest"):
        read_bundle(bundle)


def test_bundle_reader_rejects_malformed_payload_digest(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    checksums = bundle / "checksums.sha256"
    first, *rest = checksums.read_text(encoding="utf-8").splitlines()
    _, relative = first.split("  ", 1)
    checksums.write_text(
        "\n".join([f"NOT-A-SHA256  {relative}", *rest]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleReadError, match="malformed checksum"):
        read_bundle(bundle)


def test_bundle_reader_rejects_duplicate_payload_path(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    checksums = bundle / "checksums.sha256"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    checksums.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")

    with pytest.raises(BundleReadError, match="duplicate checksum path"):
        read_bundle(bundle)


@pytest.mark.parametrize(
    "relative", ["../escape", "/absolute", "runtimes\\model.bin", "."]
)
def test_bundle_reader_rejects_unsafe_payload_path(tmp_path: Path, relative: str):
    bundle = _write_bundle(tmp_path)
    checksums = bundle / "checksums.sha256"
    checksums.write_text(f"{'a' * 64}  {relative}\n", encoding="utf-8")

    with pytest.raises(BundleReadError, match="unsafe checksum path"):
        read_bundle(bundle)


def test_bundle_reader_rejects_case_colliding_payload_paths(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    checksums = bundle / "checksums.sha256"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    digest, _ = lines[0].split("  ", 1)
    checksums.write_text(
        "\n".join([*lines, f"{digest}  BUNDLE.JSON"]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleReadError, match="case-colliding checksum path"):
        read_bundle(bundle)


def test_bundle_reader_rejects_unlisted_payload(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    (bundle / "runtimes" / "mlx" / "unlisted.bin").write_bytes(b"surprise")

    with pytest.raises(BundleReadError, match="unlisted payload"):
        read_bundle(bundle)


def test_bundle_reader_rejects_manifest_mutation(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["voice_id"] = "impostor"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleReadError, match="payload checksum mismatch"):
        read_bundle(bundle)


def test_bundle_reader_rejects_symlinked_payload(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    payload = bundle / "runtimes" / "mlx" / "model.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)

    with pytest.raises(BundleReadError, match="symlink"):
        read_bundle(bundle)


def test_catalog_requires_explicit_bundle_id(tmp_path: Path):
    _write_bundle(tmp_path, bundle_id="alex@0.1.0")
    _write_bundle(tmp_path, bundle_id="sam@0.1.0")
    catalog = VoiceModelCatalog(tmp_path)

    assert [item.bundle_id for item in catalog.list()] == ["alex@0.1.0", "sam@0.1.0"]
    assert catalog.select("sam@0.1.0").voice_id == "sam"

    with pytest.raises(KeyError, match="bundle id"):
        catalog.select("latest")


def test_bundle_reader_rejects_unknown_required_schema_major(tmp_path: Path):
    bundle = _write_bundle(tmp_path)
    manifest = json.loads((bundle / "bundle.json").read_text())
    manifest["bundle_schema_version"] = "2.0.0"
    (bundle / "bundle.json").write_text(json.dumps(manifest))

    with pytest.raises(BundleReadError, match="schema"):
        read_bundle(bundle)
