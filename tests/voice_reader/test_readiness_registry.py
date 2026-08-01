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

from conftest import RUNTIME_ID, check  # the one home for these

class TestInjectedRegistry:
    """A caller may hand in the registry its composition root built.

    Without this, the registration check consults a registry nobody wired and
    reports `runtime_not_registered` no matter what the caller assembled.
    """

    def test_registry_holding_the_runtime_clears_the_blocker(self, readiness_bundle):
        """No monkeypatch: the real lister is asked about a real registry."""
        registry = RuntimeRegistry()
        registry.register(RUNTIME_ID, NullRuntime())

        report = check(readiness_bundle, registry=registry)

        assert BLOCKER_RUNTIME_NOT_REGISTERED not in report.blockers
        assert "registration" in report.checked

    def test_empty_registry_still_blocks(self, readiness_bundle):
        report = check(readiness_bundle, registry=RuntimeRegistry())

        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    def test_omitting_the_registry_preserves_current_behavior(self, readiness_bundle):
        """Regression pin: this branch registers nothing process-wide."""
        report = check(readiness_bundle)

        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    @pytest.mark.parametrize("lister_name", ["ids", "available"])
    def test_either_lister_api_is_consulted(self, readiness_bundle, lister_name):
        """The seam duck-types, so both branches' registries are answerable."""
        stand_in = type("StandIn", (), {lister_name: lambda self: (RUNTIME_ID,)})()

        report = check(readiness_bundle, registry=stand_in)

        assert BLOCKER_RUNTIME_NOT_REGISTERED not in report.blockers
