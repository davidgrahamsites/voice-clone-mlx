"""Test that the readiness check consults a caller-supplied registry.

Split out of `test_readiness.py`, which had grown past the length an ICM
contract file should ask anyone to read in one sitting. Behavior is unchanged:
the `TestInjectedRegistry` class below is the one moved verbatim.

Nothing here installs, downloads, imports a backend, or loads a model.
"""

import json

import pytest

from voiceclonegpt.synthesis.null_runtime import NullRuntime
from voiceclonegpt.synthesis.readiness import (
    BLOCKER_RUNTIME_NOT_REGISTERED,
    check_runtime_readiness,
)
from voiceclonegpt.synthesis.runtime_registry import RuntimeRegistry

RUNTIME_ID = "mlx_qwen"


@pytest.fixture
def bundle(tmp_path):
    """A bundle laid out the way `mlx_qwen_bundle` writes one.

    Duplicated from `test_readiness.py`; a shared `conftest.py` would be the
    single home, but adding one is out of scope for this split.
    """
    root = tmp_path / "alex@0.1.0"
    variant = root / "runtimes" / RUNTIME_ID
    (variant / "model").mkdir(parents=True)
    (variant / "model" / "weights.safetensors").write_bytes(b"w")
    (variant / "ref").mkdir()
    (variant / "ref" / "neutral.wav").write_bytes(b"RIFFref")
    (variant / "config.json").write_text(
        json.dumps(
            {
                "model_locator": "model",
                "ref_audio": "ref/neutral.wav",
                "ref_text": "This is the neutral reading.",
                "sample_rate": 24000,
            }
        ),
        encoding="utf-8",
    )
    (root / "bundle.json").write_text("{}", encoding="utf-8")
    return root


def check(bundle, runtime_id=RUNTIME_ID, **kwargs):
    return check_runtime_readiness(bundle, runtime_id=runtime_id, **kwargs)


class TestInjectedRegistry:
    """A caller may hand in the registry its composition root built.

    Without this, the registration check consults a registry nobody wired and
    reports `runtime_not_registered` no matter what the caller assembled.
    """

    def test_registry_holding_the_runtime_clears_the_blocker(self, bundle):
        """No monkeypatch: the real lister is asked about a real registry."""
        registry = RuntimeRegistry()
        registry.register(RUNTIME_ID, NullRuntime())

        report = check(bundle, registry=registry)

        assert BLOCKER_RUNTIME_NOT_REGISTERED not in report.blockers
        assert "registration" in report.checked

    def test_empty_registry_still_blocks(self, bundle):
        report = check(bundle, registry=RuntimeRegistry())

        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    def test_omitting_the_registry_preserves_current_behavior(self, bundle):
        """Regression pin: this branch registers nothing process-wide."""
        report = check(bundle)

        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    @pytest.mark.parametrize("lister_name", ["ids", "available"])
    def test_either_lister_api_is_consulted(self, bundle, lister_name):
        """The seam duck-types, so both branches' registries are answerable."""
        stand_in = type("StandIn", (), {lister_name: lambda self: (RUNTIME_ID,)})()

        report = check(bundle, registry=stand_in)

        assert BLOCKER_RUNTIME_NOT_REGISTERED not in report.blockers
