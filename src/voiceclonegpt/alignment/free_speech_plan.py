"""Plan transcription for free-speech candidates the overlap gate accepted.

Free Speech Mode has no script. The reader talks, a diarizer proposes
candidate spans, and this module decides — for each candidate — whether a
local transcription command should be described for it.

It owns no policy. Admission is `overlap_gate.decide_clip`; the command is
`whisper_plan.build_whisper_plan`. A second copy of either rule would drift
from the reviewed one, and the drift would be invisible until mixed-speaker
audio reached the dataset.

What this module does own is bookkeeping, and two rules make it reviewable:

**Nothing is dropped.** A rejected candidate stays in the result with
`whisper_plan=None`. A planner that returned only its accepted rows would
make "the gate refused this" and "the diarizer never proposed it"
indistinguishable, and only one of those is worth a human's attention.

**Nothing is salvaged.** When the gate refuses a candidate, the whole
candidate is refused. The clean-looking first three seconds of a clip a guest
talks over are not kept: the boundary is a diarizer's estimate, and trimming
to it trades a reviewed rejection for an unreviewed guess about where a second
voice actually starts. There is deliberately no parameter to turn this off.

Nothing here executes. The result is a description a separate runner may
later act on.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from .overlap_gate import ClipDecision, SpeakerTurn, decide_clip
from .whisper_plan import WhisperPlan, build_whisper_plan, resolve_local_audio

#: The most candidates one call will plan.
#:
#: A cap makes an unbounded diarizer result an explicit error instead of a
#: silently enormous batch of transcription work. This is roughly an hour of
#: 25-second candidates — more than one review session, which is the real
#: bottleneck downstream.
MAX_CANDIDATES = 144


class FreeSpeechPlanError(ValueError):
    """The candidate batch could not be planned."""


@dataclass(frozen=True)
class FreeSpeechCandidate:
    """One diarizer-proposed span of a free-speech recording."""

    clip_start_s: float
    clip_end_s: float
    audio_path: str
    turns: Tuple[SpeakerTurn, ...] = ()


@dataclass(frozen=True)
class PlannedCandidate:
    """What the planner decided about one candidate.

    `whisper_plan` is None exactly when `decision` is a rejection. The
    candidate's span is repeated verbatim: it is never narrowed.
    """

    clip_start_s: float
    clip_end_s: float
    audio_path: str
    decision: ClipDecision
    whisper_plan: Optional[WhisperPlan] = None


def plan_free_speech(
    candidates: Sequence[FreeSpeechCandidate],
    *,
    target_speaker: str,
    model_path: Any,
    output_dir: Any,
    language: Optional[str] = None,
) -> Tuple[PlannedCandidate, ...]:
    """Describe the transcription work for a batch of candidates.

    Args:
        candidates: Diarizer-proposed spans, at most `MAX_CANDIDATES` of them.
        target_speaker: The enrolled speaker id, passed to the overlap gate.
        model_path: Existing local model directory or weights file.
        output_dir: Existing local directory for transcripts.
        language: Optional language code for the transcription command.

    Returns:
        One frozen `PlannedCandidate` per input candidate, in input order,
        each carrying its resolved absolute `audio_path`. Accepted candidates
        carry a `WhisperPlan`; rejected ones carry None and their rejection
        reason. The input is not read again or mutated.

    Raises:
        FreeSpeechPlanError: The batch exceeds the cap, an entry is not a
            `FreeSpeechCandidate`, or the target speaker is blank.
        OverlapGateError: A candidate span or turn is malformed.
        WhisperPlanError: Any candidate's audio path — accepted or rejected —
            is missing, remote, or not a file; or an accepted candidate's
            model or output path is missing, remote, or of the wrong kind.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise FreeSpeechPlanError(f"candidates must be a sequence: {candidates!r}")

    # Measured before `tuple()`, so an oversized batch is refused without
    # being read. A lazily-loaded diarizer result should not be materialized
    # only to be thrown away — that is the work the cap exists to avoid.
    size = len(candidates)
    if size > MAX_CANDIDATES:
        raise FreeSpeechPlanError(
            f"at most {MAX_CANDIDATES} candidates per call, got {size}"
        )

    batch = tuple(candidates)

    for entry in batch:
        if not isinstance(entry, FreeSpeechCandidate):
            raise FreeSpeechPlanError(
                f"candidates must contain FreeSpeechCandidate: {entry!r}"
            )

    if isinstance(target_speaker, bool) or not isinstance(target_speaker, str):
        raise FreeSpeechPlanError(
            f"target_speaker must be a string: {target_speaker!r}"
        )
    if not target_speaker.strip():
        raise FreeSpeechPlanError("target_speaker must not be blank")

    # Every candidate's audio is validated up front — including candidates the
    # gate will reject. A remote or missing path is a fault in whatever
    # produced the batch, and letting it surface only for the candidates that
    # happened to be accepted would make the error depend on who was talking.
    audio_paths = tuple(resolve_local_audio(entry.audio_path) for entry in batch)

    planned = []
    for entry, audio in zip(batch, audio_paths):
        decision = decide_clip(
            clip_start_s=entry.clip_start_s,
            clip_end_s=entry.clip_end_s,
            target_speaker=target_speaker,
            turns=list(entry.turns),
        )
        whisper = None
        if decision.accepted:
            whisper = build_whisper_plan(
                audio,
                output_dir,
                model_path,
                language=language,
            )
        planned.append(
            PlannedCandidate(
                clip_start_s=entry.clip_start_s,
                clip_end_s=entry.clip_end_s,
                audio_path=str(audio),
                decision=decision,
                whisper_plan=whisper,
            )
        )

    return tuple(planned)
