"""File-level text-to-speech seam: text in, one audio file out.

Two modules, one job each. `backend` defines the shape every backend
satisfies and ships a stub that writes silence; `qwen_mlx` adapts a
Qwen3-TTS model running under mlx-audio to that shape.

Nothing else in the package imports this — like `bus`, it is independently
deletable, and a test asserts it stays that way. It manages no models, owns
no queue, and knows nothing about the data bus or either app.
"""

