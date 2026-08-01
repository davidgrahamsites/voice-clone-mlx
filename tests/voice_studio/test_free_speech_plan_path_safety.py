"""Free Speech planner path and batch-cap tests."""

from collections.abc import Sequence

from pathlib import Path



import pytest



from free_speech_plan_test_support import candidate, local_assets

from voiceclonegpt.alignment import free_speech_plan

from voiceclonegpt.alignment.free_speech_plan import MAX_CANDIDATES, FreeSpeechPlanError, plan_free_speech

from voiceclonegpt.alignment.overlap_gate import SpeakerTurn

from voiceclonegpt.alignment.whisper_plan import WhisperPlanError

# --- path validation is not skippable by being rejected --------------------
#
# A candidate the gate refuses still had its path recorded by whatever
# produced it. A remote path there is a bug in the caller, and discovering it
# only for the candidates that happened to be accepted would mean the error
# surfaces or hides depending on who was talking.


def test_a_remote_path_on_an_accepted_candidate_is_refused(local_assets):
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [candidate(audio_path="http://example.com/session.wav")],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


@pytest.mark.parametrize(
    "remote",
    ["http://example.com/session.wav", "//server/share/session.wav"],
)
def test_a_remote_path_on_a_rejected_candidate_is_refused(local_assets, remote):
    """The candidate below would be rejected. The path still has to be real."""
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(
                    audio_path=remote,
                    turns=[SpeakerTurn(0.2, 4.8, "guest")],
                )
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_a_missing_path_on_a_rejected_candidate_is_refused(local_assets, tmp_path):
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(
                    audio_path=tmp_path / "absent.wav",
                    turns=[SpeakerTurn(0.2, 4.8, "guest")],
                )
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_paths_are_validated_before_any_gate_call(monkeypatch, local_assets):
    monkeypatch.setattr(
        free_speech_plan,
        "decide_clip",
        lambda **_: pytest.fail("paths must be validated first"),
    )

    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [candidate(audio_path="http://example.com/session.wav")],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_a_later_candidates_bad_path_refuses_the_whole_batch(local_assets):
    """Validation is a pass over the batch, not a per-row surprise."""
    with pytest.raises(WhisperPlanError):
        plan_free_speech(
            [
                candidate(audio_path=local_assets["audio_path"]),
                candidate(audio_path="http://example.com/session.wav"),
            ],
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )


def test_path_validation_defers_to_resolve_local_audio(monkeypatch, local_assets):
    seen = []

    def fake(value):
        seen.append(value)
        return Path(local_assets["audio_path"]).resolve()

    monkeypatch.setattr(free_speech_plan, "resolve_local_audio", fake)

    plan_free_speech(
        [
            candidate(audio_path="anything-at-all.wav"),
            candidate(
                audio_path="also-anything.wav",
                turns=[SpeakerTurn(0.2, 4.8, "guest")],
            ),
        ],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    assert seen == ["anything-at-all.wav", "also-anything.wav"]


# --- the cap is enforced before the batch is materialized ------------------


class OversizedSequence(Sequence):
    """Reports a length over the cap and refuses to be read.

    A planner that measured `len(tuple(candidates))` would have to consume the
    sequence first. For a generator-backed or lazily-loaded diarizer result
    that is exactly the work the cap exists to avoid.
    """

    def __len__(self):
        return MAX_CANDIDATES + 1

    def __getitem__(self, index):
        raise AssertionError("an oversized batch must not be read")

    def __iter__(self):
        raise AssertionError("an oversized batch must not be iterated")


def test_an_oversized_sequence_is_never_iterated(local_assets):
    with pytest.raises(FreeSpeechPlanError):
        plan_free_speech(
            OversizedSequence(),
            target_speaker="owner",
            model_path=local_assets["model_path"],
            output_dir=local_assets["output_dir"],
        )
