"""Answer whether a real model round trip could run here, and if not, why.

One job: report. It installs nothing, downloads nothing, imports no backend,
loads no model, and writes no file. Given a bundle and a runtime id it names
**every** thing standing between this machine and real synthesis — all of them
at once, so the answer is a checklist rather than a first-failure.

Nothing is validated twice. Bundle validity is `shared.bundle_reader`'s answer,
registration is `runtime_registry`'s, and config validity is obtained by
running `MlxQwenRuntime.load` with a **probe loader that raises before any
model is opened**. Dependency presence uses `importlib.util.find_spec`, which
locates a package without importing it.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from voiceclonegpt.synthesis.mlx_qwen_runtime import (
    MlxQwenRuntime,
    RuntimeConfigError,
)
try:  # pragma: no cover - one branch per checkout
    from voiceclonegpt.synthesis.runtime_registry import default_registry
except ImportError:  # pragma: no cover
    #: `main`'s registry has no shared instance and no `default_registry`.
    #: Registration there is done explicitly by a composition root, so there
    #: is no process-wide registry for this module to consult.
    default_registry = None

from voiceclonegpt.synthesis.runtime_registry import RuntimeRegistry

try:  # pragma: no cover - exercised in whichever checkout has the module
    from voiceclonegpt.shared.bundle_reader import read_bundle
except ImportError:  # pragma: no cover
    #: `shared/bundle_reader.py` ships on the model-roundtrip branch. Where it
    #: is absent the bundle check cannot run; it is then omitted from
    #: `checked` rather than guessed at.
    read_bundle = None

#: The backend package a real Qwen round trip needs. Located, never imported.
DEPENDENCY = "mlx_audio"

BLOCKER_DEPENDENCY_MISSING = "dependency_missing"
BLOCKER_RUNTIME_NOT_REGISTERED = "runtime_not_registered"
BLOCKER_BUNDLE_UNREADABLE = "bundle_unreadable"
BLOCKER_CONFIG_INVALID = "config_invalid"
BLOCKER_MODEL_ABSENT = "model_absent"
BLOCKER_REFERENCE_ABSENT = "reference_absent"

#: The complete vocabulary. Fixed and small: free-text reasons cannot be
#: filtered on, and two reports would not be comparable.
BLOCKERS = (
    BLOCKER_BUNDLE_UNREADABLE,
    BLOCKER_CONFIG_INVALID,
    BLOCKER_DEPENDENCY_MISSING,
    BLOCKER_MODEL_ABSENT,
    BLOCKER_REFERENCE_ABSENT,
    BLOCKER_RUNTIME_NOT_REGISTERED,
)

#: Every check that must have run before a report may claim readiness.
REQUIRED_CHECKS = ("dependency", "registration", "bundle", "config")

CONFIG_NAME = "config.json"


class ProbeReached(Exception):
    """Raised by the default probe when config validation has passed.

    Reaching the loader is the signal that the config is usable. The probe
    raises instead of returning, so no model is ever opened.
    """


@dataclass(frozen=True)
class ReadinessReport:
    """What stands between this machine and a real round trip.

    `ready` is True only when nothing blocked **and** every required check
    actually ran. An empty `blockers` with `ready` False means something could
    not be verified — which is not the same as being fine.
    """

    ready: bool
    blockers: Tuple[str, ...] = ()
    checked: Tuple[str, ...] = ()


def _dependency_present(name: str) -> bool:
    """True if `name` can be located, without importing it.

    `find_spec` reads the import system's metadata only. Importing `mlx_audio`
    here would pull a heavy backend into a process that only wants to answer a
    question.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _registered_runtimes(registry=None) -> Tuple[str, ...]:
    """The runtime ids a caller could actually select.

    The detached checkout exposes a shared ``default_registry().available``;
    the main checkout exposes ``RuntimeRegistry.ids`` and no shared instance.
    Ask whichever API exists rather than duplicating registration here.

    Args:
        registry: A registry a composition root already built. When ``None``,
            fall back to whatever this checkout exposes process-wide.
    """
    if registry is None:
        registry = (
            default_registry() if default_registry is not None else RuntimeRegistry()
        )
    lister = getattr(registry, "available", None) or getattr(registry, "ids")
    return tuple(lister())


def _probe(locator):
    """The loader stand-in. Never returns a model — that is the point.

    Private, and not overridable from the public signature: an injectable
    probe would let a caller hand in something that genuinely opens a model,
    which is the one thing this module promises never to do.
    """
    raise ProbeReached(locator)


def _classify_config_error(exc: RuntimeConfigError) -> str:
    """Map the runtime's own refusal onto the blocker vocabulary.

    The runtime names the offending config field in its message, and those
    field names are part of the config contract rather than incidental prose.
    Anything unrecognized stays the general `config_invalid` rather than being
    guessed at.
    """
    message = str(exc)

    if "model_locator" in message:
        return BLOCKER_MODEL_ABSENT
    if "ref_audio" in message:
        return BLOCKER_REFERENCE_ABSENT
    return BLOCKER_CONFIG_INVALID


def check_runtime_readiness(
    bundle_dir, *, runtime_id: str, registry=None
) -> ReadinessReport:
    """Report everything preventing a real round trip for one bundle.

    Args:
        bundle_dir: The bundle to check. Never modified.
        runtime_id: The runtime variant a caller would select.
        registry: The registry a composition root built, if the caller has one.
            Omitted, the check consults whatever this checkout exposes
            process-wide — which on a branch with no shared registry is empty,
            so every id reports `runtime_not_registered`.

    There is deliberately **no** `probe` parameter. Config validity is
    established by reaching `_probe`, which always raises — so a readiness
    check can never be turned into a model load by passing a loader that
    returns one. A supplied `registry` is only ever asked for its ids; it
    cannot reach the probe or otherwise weaken that guarantee.

    Returns:
        A `ReadinessReport`. Blockers are sorted and unique, and every code is
        from `BLOCKERS`.
    """
    blockers = set()
    checked = []

    # 1. Backend package — located, not imported.
    checked.append("dependency")
    if not _dependency_present(DEPENDENCY):
        blockers.add(BLOCKER_DEPENDENCY_MISSING)

    # 2. Could a caller even select this runtime?
    checked.append("registration")
    # Called with no argument when nothing was injected, so a test that
    # substitutes a zero-argument lister still sees the call it expects.
    ids = _registered_runtimes() if registry is None else _registered_runtimes(registry)
    if runtime_id not in ids:
        blockers.add(BLOCKER_RUNTIME_NOT_REGISTERED)

    bundle_dir = Path(bundle_dir)

    # 3. Bundle validity is the reader's answer, not ours.
    if read_bundle is not None:
        checked.append("bundle")
        try:
            read_bundle(bundle_dir)
        except Exception:
            blockers.add(BLOCKER_BUNDLE_UNREADABLE)

    # 4. Config validity comes from the runtime's own validation. Reaching the
    #    probe means every rule passed; no model is opened either way.
    checked.append("config")
    config_path = bundle_dir / "runtimes" / runtime_id / CONFIG_NAME
    try:
        MlxQwenRuntime(loader=_probe).load(config_path, {})
    except ProbeReached:
        pass
    except RuntimeConfigError as exc:
        blockers.add(_classify_config_error(exc))
    except Exception:
        blockers.add(BLOCKER_CONFIG_INVALID)

    ready = not blockers and set(checked) == set(REQUIRED_CHECKS)

    return ReadinessReport(
        ready=ready,
        blockers=tuple(sorted(blockers)),
        checked=tuple(checked),
    )
