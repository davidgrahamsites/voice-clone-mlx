"""Free Speech planner inertness tests."""

import ast

from pathlib import Path



from free_speech_plan_test_support import candidate, local_assets

from voiceclonegpt.alignment import free_speech_plan

from voiceclonegpt.alignment.free_speech_plan import plan_free_speech



SOURCE = Path(free_speech_plan.__file__).read_text(encoding="utf-8")

TREE = ast.parse(SOURCE)

# --- inertness -------------------------------------------------------------


FORBIDDEN_IMPORTS = {
    "subprocess",
    "os.system",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "mlx",
    "mlx_whisper",
    "whisper",
    "torch",
    "numpy",
    "soundfile",
    "librosa",
    "asyncio",
    "multiprocessing",
    "shutil",
}


def imported_names():
    names = set()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_no_process_network_or_model_imports():
    for name in imported_names():
        root = name.split(".")[0]
        assert root not in FORBIDDEN_IMPORTS, name


def test_no_forbidden_call_names_appear():
    called = {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    for forbidden in (
        "run",
        "call",
        "check_output",
        "Popen",
        "system",
        "spawn",
        "urlopen",
        "eval",
        "exec",
        "compile",
        "open",
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
    ):
        assert forbidden not in called


def test_planning_writes_no_file(local_assets):
    output = local_assets["output_dir"]
    before = sorted(p.name for p in output.iterdir())

    plan_free_speech(
        [candidate(audio_path=local_assets["audio_path"])],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=output,
    )

    assert sorted(p.name for p in output.iterdir()) == before


def test_the_planner_returns_an_argv_it_does_not_run(local_assets):
    results = plan_free_speech(
        [candidate(audio_path=local_assets["audio_path"])],
        target_speaker="owner",
        model_path=local_assets["model_path"],
        output_dir=local_assets["output_dir"],
    )

    argv = results[0].whisper_plan.argv
    assert isinstance(argv, tuple)
    assert "-m" in argv


