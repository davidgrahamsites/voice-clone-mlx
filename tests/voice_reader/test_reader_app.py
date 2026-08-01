"""Playback, window, and headless-display tests for the Voice Reader shell.

Manifest schema and audio-path security live in
`test_reader_manifest_security.py`; this file covers what the app *does* with a
manifest once it is loaded.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from voiceclonegpt.reader_app.core import Player, Take, load_manifest


#: Real Tk windows are only built when this is explicitly set to "1".
UI_TESTS_ENV = "VOICECLONEGPT_RUN_UI_TESTS"

#: Passed to `_open_hidden_root` by tests that drive the post-gate branches.
OPTED_IN = {UI_TESTS_ENV: "1"}

DISPLAY_ERROR_MARKERS = (
    "no display",
    "display name",
    "couldn't connect to display",
    "can't find a usable init.tcl",
    "can't find package tk",
    "application-specific initialization failed",
    "unable to load tk",
)


def _is_no_display_error(exc) -> bool:
    """True when a TclError means "no display", not "the widget is broken"."""
    message = str(exc).lower()
    return any(marker in message for marker in DISPLAY_ERROR_MARKERS)


def _ui_tests_enabled(env) -> bool:
    """True only when the caller has explicitly opted into real UI tests."""
    return env.get(UI_TESTS_ENV) == "1"


def _open_hidden_root(factory=None, env=None):
    """Open a withdrawn Tk root, or skip before Tk is ever touched.

    The opt-in gate runs **first and unconditionally**. In a headless runner
    Tk can abort the process rather than raise, and an abort cannot be caught
    by any `except` clause — so the only safe guard is never to initialize Tk
    unless the caller has asked for it.

    The one place `tkinter.Tk` is called in this module. `factory` and `env`
    are injected by the tests below so every branch is exercised on a machine
    that does have a display.
    """
    env = os.environ if env is None else env
    if not _ui_tests_enabled(env):
        pytest.skip(
            f"UI tests are opt-in; run with {UI_TESTS_ENV}=1 to execute them"
        )

    tk = pytest.importorskip("tkinter")
    if factory is None:
        factory = tk.Tk

    try:
        root = factory()
    except tk.TclError as exc:
        if not _is_no_display_error(exc):
            raise
        pytest.skip(f"Tk has no display: {exc}")

    root.withdraw()
    return root


@pytest.fixture
def reader_tk_root():
    """A hidden Tk root for Voice Reader widget tests.

    Each app owns its own headless-display seam rather than sharing one:
    Voice Reader must stay independently deletable, so its tests may not
    depend on a fixture that lives with the other app. The name is
    app-specific on purpose — an earlier version called this `tk_root`, which
    silently shadowed the permissive fixture in `tests/conftest.py`, so a typo
    or a deleted fixture would have fallen back to it unnoticed. Under this
    name a missing fixture is an error, not a quiet downgrade.

    Skips only on a recognized display failure; any other TclError is
    re-raised so a real widget bug cannot hide as an environment skip.
    """
    root = _open_hidden_root()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def manifest(tmp_path):
    records = [
        {"id": "NEUTRAL-001", "text": "First line.", "style": "neutral"},
        {"id": "NEUTRAL-002", "text": "Second line.", "style": "neutral"},
    ]
    path = tmp_path / "script_session_1.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (tmp_path / "NEUTRAL-001.wav").write_bytes(b"")
    return path


class TestPlayer:
    def test_playing_a_take_without_audio_reports_it(self):
        player = Player()
        message = player.play(Take(id="X-1", text="hi", style="neutral"))
        assert "No audio" in message
        assert not player.is_playing

    def test_missing_player_binary_is_reported_not_raised(self, manifest):
        player = Player(command="/nonexistent/afplay")
        message = player.play(load_manifest(manifest)[0])
        assert "Could not play" in message

    def test_stop_is_safe_when_nothing_is_playing(self):
        Player().stop()


class TestReaderWindow:
    def test_window_builds_and_lists_takes(self, reader_tk_root, manifest):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        window = ReaderWindow(reader_tk_root)
        assert str(window.play_button["state"]) == "disabled"

        window.load_path(manifest)
        assert window.take_list.size() == 2
        assert str(window.play_button["state"]) == "normal"
        assert "1 with audio" in window.status_var.get()

    def test_load_error_shows_in_status_bar(self, reader_tk_root, tmp_path):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        window = ReaderWindow(reader_tk_root)
        window.load_path(tmp_path / "missing.jsonl")
        assert "not found" in window.status_var.get().lower()

    def test_play_without_selection_prompts(self, reader_tk_root, manifest):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        window = ReaderWindow(reader_tk_root)
        window.load_path(manifest)
        window.on_play()
        assert "Select a take" in window.status_var.get()

    def test_open_uses_the_file_picker_seam(self, reader_tk_root, manifest):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        window = ReaderWindow(reader_tk_root, choose_file=lambda: str(manifest))
        window.on_open()
        assert window.take_list.size() == 2

    def test_play_uses_the_injected_player(self, reader_tk_root, manifest):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        played = []

        class FakePlayer:
            def play(self, take):
                played.append(take.id)
                return f"Playing {take.id}."

            def stop(self):
                pass

        window = ReaderWindow(reader_tk_root, player=FakePlayer())
        window.load_path(manifest)
        window.take_list.selection_set(0)
        window.on_play()
        assert played == ["NEUTRAL-001"]

    def test_reader_does_not_import_the_studio(self):
        import voiceclonegpt.reader_app.core as core
        import voiceclonegpt.reader_app.ui as ui

        for module in (core, ui):
            source = open(module.__file__, encoding="utf-8").read()
            assert "studio_app" not in source

    def test_generate_enables_only_after_ready_voice_and_text(
        self, reader_tk_root, tmp_path
    ):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        calls = []

        class FakeController:
            ready = False

            def inspect_bundle(self, path):
                calls.append(("inspect", Path(path)))
                return SimpleNamespace(
                    bundle_id="alex@0.7.0", runtime_ids=("fake_voice",)
                )

            def select(self, path, runtime_id):
                calls.append(("select", Path(path), runtime_id))
                self.ready = True
                return SimpleNamespace(ready=True, blockers=())

            def set_text(self, text):
                calls.append(("text", text))

            def synthesize(self, output_dir, output_name):
                calls.append(("synthesize", Path(output_dir), output_name))
                return Path(output_dir) / output_name

        controller = FakeController()
        window = ReaderWindow(
            reader_tk_root,
            synthesis_controller=controller,
            choose_bundle=lambda: str(tmp_path / "alex"),
            choose_output_dir=lambda: str(tmp_path),
        )

        assert str(window.generate_button["state"]) == "disabled"
        window.on_choose_bundle()
        assert tuple(window.runtime_box["values"]) == ("fake_voice",)
        assert str(window.generate_button["state"]) == "disabled"

        window.runtime_var.set("fake_voice")
        window.on_check_voice()
        assert str(window.generate_button["state"]) == "disabled"

        window.synthesis_text.insert("1.0", "Speak this exact line.")
        window._sync_generate_state()
        assert str(window.generate_button["state"]) == "normal"
        window.on_generate()

        assert calls == [
            ("inspect", tmp_path / "alex"),
            ("select", tmp_path / "alex", "fake_voice"),
            ("text", "Speak this exact line."),
            ("synthesize", tmp_path, "reader-output.wav"),
        ]

    def test_text_picker_ingests_through_the_controller(
        self, reader_tk_root, tmp_path
    ):
        from voiceclonegpt.reader_app.ui import ReaderWindow

        text_path = tmp_path / "chapter.txt"

        class FakeController:
            ready = False

            def ingest_text(self, path):
                assert Path(path) == text_path
                return "Imported exact text."

        window = ReaderWindow(
            reader_tk_root,
            synthesis_controller=FakeController(),
            choose_text_file=lambda: str(text_path),
        )

        window.on_open_text()

        assert window.synthesis_text.get("1.0", "end-1c") == "Imported exact text."


class TestHeadlessDisplaySeam:
    """The skip path must actually work and must not hide real failures.

    This machine has a display, so both branches are driven through the
    injected factory rather than by unsetting the environment.
    """

    def test_skips_when_the_opt_in_is_absent(self):
        with pytest.raises(pytest.skip.Exception, match=UI_TESTS_ENV):
            _open_hidden_root(env={})

    @pytest.mark.parametrize("value", ["", "0", "no", "false", "TRUE", "yes"])
    def test_only_the_exact_opt_in_value_enables_ui_tests(self, value):
        with pytest.raises(pytest.skip.Exception):
            _open_hidden_root(env={UI_TESTS_ENV: value})

    def test_gate_runs_before_tk_is_constructed(self):
        """A headless runner can abort inside Tk, so never reach it."""
        built = []

        def recording_factory():
            built.append(True)
            raise AssertionError("Tk must not be constructed without opt-in")

        with pytest.raises(pytest.skip.Exception):
            _open_hidden_root(factory=recording_factory, env={})

        assert built == []

    def test_opt_in_lets_construction_proceed(self):
        """With the opt-in set, real widget failures still surface."""

        def broken():
            raise AssertionError("widget bug")

        with pytest.raises(AssertionError, match="widget bug"):
            _open_hidden_root(factory=broken, env=OPTED_IN)

    def _raiser(self, message):
        tk = pytest.importorskip("tkinter")

        def factory():
            raise tk.TclError(message)

        return factory

    @pytest.mark.parametrize(
        "message",
        [
            "no display name and no $DISPLAY environment variable",
            "couldn't connect to display :0",
            "Can't find a usable init.tcl in the following directories",
            "can't find package Tk",
            "application-specific initialization failed: unable to load Tk",
        ],
    )
    def test_skips_on_a_known_no_display_error(self, message):
        with pytest.raises(pytest.skip.Exception):
            _open_hidden_root(factory=self._raiser(message), env=OPTED_IN)

    def test_skip_message_names_the_cause(self):
        with pytest.raises(pytest.skip.Exception, match="no display name"):
            _open_hidden_root(
                factory=self._raiser("no display name and no $DISPLAY"),
                env=OPTED_IN,
            )

    def test_reraises_an_unrelated_tcl_error(self):
        """A real widget bug must not be reported as an environment skip."""
        tk = pytest.importorskip("tkinter")

        with pytest.raises(tk.TclError, match="invalid command name"):
            _open_hidden_root(
                factory=self._raiser('invalid command name ".!b"'), env=OPTED_IN
            )

    def test_reraises_a_non_tcl_error(self):
        def boom():
            raise RuntimeError("something else entirely")

        with pytest.raises(RuntimeError, match="something else"):
            _open_hidden_root(factory=boom, env=OPTED_IN)

    def test_returns_a_withdrawn_root(self):
        withdrawn = []

        class FakeRoot:
            def withdraw(self):
                withdrawn.append(True)

        root = _open_hidden_root(factory=FakeRoot, env=OPTED_IN)

        assert isinstance(root, FakeRoot)
        assert withdrawn == [True]

    def test_is_the_only_place_tk_is_constructed(self):
        """No test may call tkinter.Tk() outside the fixture path."""
        source = Path(__file__).read_text(encoding="utf-8")
        # Needles are built at runtime so these assertions are not themselves
        # matches in the file they scan.
        call = "tk.Tk" + "()"
        default = "factory = tk." + "Tk"

        assert call not in source
        assert source.count(default) == 1
