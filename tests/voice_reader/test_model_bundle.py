import hashlib
import json
from pathlib import Path

import pytest

from voiceclonegpt.shared.bundle_reader import BundleReadError, read_bundle
from voiceclonegpt.shared.model_bundle import ArtifactKind
from voiceclonegpt.shared.voice_model_catalog import VoiceModelCatalog


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
            }
        )
    )
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
