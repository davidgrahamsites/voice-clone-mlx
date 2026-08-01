"""Manifest and runner-result validation tests for the CUDA command provider."""

import pytest

from voiceclonegpt.training.cuda_command_provider import (
    CudaCommandTrainingProvider, InvalidCommandManifest, InvalidCommandResult,
)
from cuda_command_provider_test_support import RecordingRunner, approved, command_result, manifest, request


def test_runner_must_return_a_command_result():
    class MalformedRunner:
        def run(self, plan):
            return object()
    provider = CudaCommandTrainingProvider(manifest(), approved(), MalformedRunner())
    with pytest.raises(InvalidCommandResult, match="CommandResult"):
        provider.train(request(), resume_from=None)


@pytest.mark.parametrize(("field", "value"), (("timeout_seconds", 0), ("timeout_seconds", True), ("max_output_bytes", 0), ("max_output_bytes", True), ("argv_prefix", "python -m train"), ("argv_prefix", ())))
def test_command_limits_and_argv_are_validated(field, value):
    with pytest.raises(InvalidCommandManifest, match=field):
        manifest(**{field: value})


@pytest.mark.parametrize(("field", "value"), (("exit_code", True), ("captured_output_bytes", -1), ("checkpoint_id", " "), ("checkpoint_path", ""), ("checkpoint_sha256", "bad"), ("artifact_kind", "reference_clone")))
def test_malformed_runner_fields_are_typed_failures(field, value):
    provider = CudaCommandTrainingProvider(manifest(), approved(), RecordingRunner(command_result(**{field: value})))
    with pytest.raises(InvalidCommandResult, match=field):
        provider.train(request(), resume_from=None)


@pytest.mark.parametrize("field", ("recipe_id", "recipe_revision", "backend_id", "working_directory", "dataset_manifest_path", "training_config_path", "checkpoint_directory", "target_id"))
def test_command_manifest_requires_all_identity_and_path_text(field):
    with pytest.raises(InvalidCommandManifest, match=field):
        manifest(**{field: "  "})


@pytest.mark.parametrize("environment", ("CUDA_VISIBLE_DEVICES=0", (("", "0"),), (("CUDA_VISIBLE_DEVICES", 0),), (("CUDA_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", "1"))))
def test_environment_allowlist_is_explicit_and_unambiguous(environment):
    with pytest.raises(InvalidCommandManifest, match="environment"):
        manifest(environment=environment)
