from dataclasses import dataclass
from enum import Enum
from typing import Any


class ArtifactKind(str, Enum):
    REFERENCE_CLONE = "reference_clone"
    FINE_TUNED_FULL = "fine_tuned_full"
    FINE_TUNED_ADAPTER = "fine_tuned_adapter"


@dataclass(frozen=True)
class VoiceModelBundle:
    bundle_id: str
    voice_id: str
    model_version: str
    artifact_kind: ArtifactKind
    manifest: dict[str, Any]

