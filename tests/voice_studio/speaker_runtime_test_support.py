from voiceclonegpt.alignment.speaker_runtime import RuntimeProvenance


def provenance():
    return RuntimeProvenance(
        diarizer="mlx_sortformer",
        diarizer_version="local-r1",
        verifier="speechbrain_ecapa",
        verifier_version="local-r2",
        enrollment_id="owner-enrollment-v1",
    )
