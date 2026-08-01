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

from conftest import RUNTIME_ID, check  # the one home for these


class TestProbeIsNotOverridable:
    """The probe is private, and that is the whole guarantee.

    A public `probe=` would let a caller pass a loader that genuinely opens a
    model — turning a readiness *check* into a model load, which is the one
    thing this module promises never to do.
    """

    def test_there_is_no_public_probe_parameter(self):
        parameters = inspect.signature(check_runtime_readiness).parameters

        assert "probe" not in parameters

    def test_passing_a_probe_is_rejected(self, readiness_bundle, all_clear):
        with pytest.raises(TypeError):
            check_runtime_readiness(
                readiness_bundle, runtime_id=RUNTIME_ID, probe=lambda locator: "a model"
            )

    def test_the_default_probe_never_returns(self, readiness_bundle):
        with pytest.raises(readiness.ProbeReached):
            readiness._probe("any-locator")

    def test_a_valid_config_reaches_the_probe(self, readiness_bundle, all_clear):
        """Config validity is established by reaching the loader, not past it."""
        report = check(readiness_bundle)

        assert "config" in report.checked
        assert BLOCKER_CONFIG_INVALID not in report.blockers
