from pathlib import Path

from .bundle_reader import read_bundle
from .model_bundle import VoiceModelBundle


class VoiceModelCatalog:
    """Lists and explicitly selects verified bundles without loading weights."""

    def __init__(self, root: Path):
        self._root = root

    def list(self) -> list[VoiceModelBundle]:
        bundles = []
        for manifest in sorted(self._root.glob("*/bundle.json")):
            bundles.append(read_bundle(manifest.parent))
        return bundles

    def select(self, bundle_id: str) -> VoiceModelBundle:
        if bundle_id == "latest":
            raise KeyError("bundle id must be explicit; latest is not allowed")
        for bundle in self.list():
            if bundle.bundle_id == bundle_id:
                return bundle
        raise KeyError(f"unknown bundle id: {bundle_id}")

