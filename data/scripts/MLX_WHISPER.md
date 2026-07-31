# Local MLX Whisper integration

The machine currently has Whisper resources in these locations:

- `/Users/appleadmin/.cache/whisper/large-v3.pt`
- `/Users/appleadmin/.cache/whisper/tiny.pt`
- `/Users/appleadmin/miniforge3/envs/whisperflow/`
- `/Users/appleadmin/miniforge3/envs/videowhisper/`
- `/Users/appleadmin/Apps/WhisperFlow/WhisperFlow V2.app/Contents/Resources/mlx_whisper/`

Voice Studio should discover these paths rather than downloading another copy.
The MLX Whisper runtime is used only for speech-to-text and timestamps. It does
not synthesize the personal voice; Voice Reader will call the separately
versioned text-to-speech model for that step.

The integration should expose a configurable command/runtime adapter because
the bundled application may change its internal invocation format. It should
record the selected runtime, model path, model identifier, and command version
in every transcription manifest for reproducibility.

