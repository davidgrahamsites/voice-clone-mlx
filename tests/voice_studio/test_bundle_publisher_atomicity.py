"""Bundle publisher atomicity and seam tests."""

import ast
from pathlib import Path

import pytest

from bundle_publisher_test_support import FakeCopier, request
from voiceclonegpt.training.bundle_publisher import BundlePublicationError, publish_bundle


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
