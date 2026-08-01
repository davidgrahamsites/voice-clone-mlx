# Recording scripts

- [`voice_training_script_30_minutes.md`](voice_training_script_30_minutes.md)
  is the ready-to-read two-session pilot used before any real training spend.
- [`pilot_recording_script.md`](pilot_recording_script.md) is the shorter
  recording-chain check.

Voice Studio stores generated recording scripts here. Each script should have:

- a human-readable Markdown teleprompter file;
- a machine-readable JSONL manifest;
- a stable utterance ID for every reading prompt;
- a canonical style for every block;
- session and block boundaries;
- calibration passages repeated at the beginning and end of each session; and
- source provenance indicating whether a prompt came from owned material or was
  generated to fill a coverage gap.

## Spoken block format

Each block follows this pattern:

```text
This is the warm reading.

[Pause for three seconds.]

WARM-001: It was good to hear from you after such a long and eventful week.

[Pause for two seconds.]

WARM-002: Take all the time you need; there is no hurry at all.
```

The utterance IDs are displayed but are not spoken. The recorder says the style
marker once, pauses, reads each prompt, and leaves a short silence between
prompts. MLX Whisper timestamps and the known script are used together to align
the recording locally on Apple Silicon. Marker audio is retained in the raw
session but excluded from model training clips.

## Canonical styles

- `neutral`
- `warm`
- `energetic`
- `serious`
- `somber`
- `questioning`
- `emphasis`
- `dialogue`

The spoken phrase "This is the netural reading" is accepted as an alias, but
all resulting metadata and folders use `neutral`.

## Script composition

The full script should target approximately 60–70% neutral narration. The
remaining material is distributed deliberately among the other styles. Some
prompts should be repeated across styles so the model sees the same words with
different delivery, while most prompts remain unique to maximize linguistic
coverage.

The generator should measure and report coverage for letters and phonemes,
questions, exclamations, dialogue, short and long sentences, names, acronyms,
dates, times, currency, decimals, percentages, URLs, and domain-specific terms.
