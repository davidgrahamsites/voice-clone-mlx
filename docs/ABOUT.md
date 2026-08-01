# About VoiceCloneGPT

## Short description

VoiceCloneGPT is a private, local-first macOS workspace for creating an
expressive voice model from the owner's recordings and using that model to
read the owner's writing aloud. **Voice Studio** records, transcribes, aligns,
and rejects contaminated clips. **Voice Reader** imports reviewed text and
generates resumable audio from an explicitly verified voice bundle.

## The promise

Your recordings stay under your control. The project is designed around clean
single-speaker evidence, deliberate expressive styles, reproducible manifests,
versioned model bundles, and a clear separation between the app that produces a
voice and the app that consumes it.

## The boundary

This is a personal, non-commercial utility—not a hosted voice service. It does
not upload private data by default, accept a clip with overlapping speakers, or
call a silent placeholder a trained model. The repository's software contracts
and tests are substantially built; the final provider installation, model
training/conversion, and audible real-model verification remain explicit gates.

## Suggested first run

1. Read the [real round-trip procedure](REAL_VOICE_ROUNDTRIP_PROCEDURE.md).
2. Record the [30-minute pilot script](../data/scripts/voice_training_script_30_minutes.md).
3. Inspect the masters, transcripts, rejection decisions, and checksums.
4. Run the bounded backend smoke test before spending GPU time.
5. Promote a model only after a human listens to held-out prompts and confirms
   that the output is non-silent, intelligible, expressive, and recognizably
   the intended speaker.

For the complete repository map and commands, see the [README](../README.md).
