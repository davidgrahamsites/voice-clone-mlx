"""Test that readiness can never load a model.

The probe is the guarantee: config validity is established by *reaching* the
loader, and the loader stand-in always raises. It is private and not
overridable, so a caller cannot hand in something that genuinely opens a
model and turn a readiness check into a model load.

Everything else about the check — blockers, the report contract, purity —
lives in `test_readiness.py`.
"""

import inspect
import json

import pytest

from voiceclonegpt.synthesis import readiness
from voiceclonegpt.synthesis.readiness import (
    BLOCKER_CONFIG_INVALID,
    check_runtime_readiness,
)


RUNTIME_ID = "mlx_qwen"


@pytest.fixture
def bundle(tmp_path):
    """A bundle laid out the way `mlx_qwen_bundle` writes one."""
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


@pytest.fixture
def all_clear(monkeypatch):
    """Make every environment-dependent check pass."""
    monkeypatch.setattr(readiness, "_dependency_present", lambda name: True)
    monkeypatch.setattr(readiness, "_registered_runtimes", lambda: (RUNTIME_ID,))
    monkeypatch.setattr(readiness, "read_bundle", lambda path: object())


def check(bundle, runtime_id=RUNTIME_ID, **kwargs):
    return check_runtime_readiness(bundle, runtime_id=runtime_id, **kwargs)


class TestProbeIsNotOverridable:
    """The probe is private, and that is the whole guarantee.

    A public `probe=` would let a caller pass a loader that genuinely opens a
    model — turning a readiness *check* into a model load, which is the one
    thing this module promises never to do.
    """

    def test_there_is_no_public_probe_parameter(self):
        parameters = inspect.signature(check_runtime_readiness).parameters

        assert "probe" not in parameters

    def test_passing_a_probe_is_rejected(self, bundle, all_clear):
        with pytest.raises(TypeError):
            check_runtime_readiness(
                bundle, runtime_id=RUNTIME_ID, probe=lambda locator: "a model"
            )

    def test_the_default_probe_never_returns(self, bundle):
        with pytest.raises(readiness.ProbeReached):
            readiness._probe("any-locator")

    def test_a_valid_config_reaches_the_probe(self, bundle, all_clear):
        """Config validity is established by reaching the loader, not past it."""
        report = check(bundle)

        assert "config" in report.checked
        assert BLOCKER_CONFIG_INVALID not in report.blockers
