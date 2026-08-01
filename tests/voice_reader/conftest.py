"""Shared fixtures for the readiness tests.

`test_readiness.py`, `test_readiness_registry.py`, and `test_readiness_probe.py`
were split apart for length and each carried its own copy of the same bundle
layout, the same monkeypatch set, and the same `check` helper. Three copies of
one fact can drift independently; this is the single home.

**The bundle fixture is `readiness_bundle`, not `bundle`, on purpose.**
`test_mlx_qwen_runtime.py`, `test_mlx_qwen_runtime_config.py`, and
`test_mlx_qwen_audio_conversion.py` already define a module-local `bundle` in
this same directory, and theirs is a *different* layout — rooted at
`tmp_path/"runtime"` with no `bundle.json` and no `runtimes/<id>/` tree. A
fixture named `bundle` here would be silently shadowed by each of them, with no
error or warning, and would quietly become wrong the day one of those local
copies was deleted in the belief that the shared one would take over.

Nothing here installs, downloads, imports a backend, or loads a model.
"""

import json

import pytest

from voiceclonegpt.synthesis import readiness
from voiceclonegpt.synthesis.readiness import check_runtime_readiness

#: The runtime variant every readiness test asks about.
RUNTIME_ID = "mlx_qwen"


@pytest.fixture
def readiness_bundle(tmp_path):
    """A bundle laid out the way `mlx_qwen_bundle` writes one."""
    root = tmp_path / "alex@0.1.0"
    variant = root / "runtimes" / RUNTIME_ID
    (variant / "model").mkdir(parents=True)
    (variant / "model" / "weights.safetensors").write_bytes(b"w")
    (variant / "ref").mkdir()
    (variant / "ref" / "neutral.wav").write_bytes(b"RIFFref")
    (variant / "config.json").write_text(
        json.dumps(
            {
                "model_locator": "model",
                "ref_audio": "ref/neutral.wav",
                "ref_text": "This is the neutral reading.",
                "sample_rate": 24000,
            }
        ),
        encoding="utf-8",
    )
    (root / "bundle.json").write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def all_clear(monkeypatch):
    """Make every environment-dependent check pass."""
    monkeypatch.setattr(readiness, "_dependency_present", lambda name: True)
    monkeypatch.setattr(readiness, "_registered_runtimes", lambda: (RUNTIME_ID,))
    monkeypatch.setattr(readiness, "read_bundle", lambda path: object())


def check(bundle, runtime_id=RUNTIME_ID, **kwargs):
    """Call the seam with the usual runtime id.

    A plain function rather than a fixture: the three modules import it, and a
    helper that takes its subject as an argument reads better at each call site
    than one that has to be requested first.
    """
    return check_runtime_readiness(bundle, runtime_id=runtime_id, **kwargs)
