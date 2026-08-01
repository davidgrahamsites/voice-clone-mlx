"""Smoke tests for the Voice Studio app shell."""

import os
from pathlib import Path

import pytest

from voiceclonegpt.shared import integration_seam
from voiceclonegpt.studio_app.core import StudioSession


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
def studio_tk_root():
    """A hidden Tk root for Voice Studio widget tests.

    Each app owns its own headless-display seam rather than sharing one:
    Voice Studio must stay independently deletable, so its tests may not
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


SAMPLE = "It was good to hear from you. Take all the time you need."


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class TestStudioSession:
    def test_load_script_splits_into_lines(self, source_file):
        script = StudioSession().load_script(source_file)
        assert script.path == source_file
        assert script.lines == [
            "It was good to hear from you.",
            "Take all the time you need.",
        ]

    def test_load_script_rejects_unsupported_format(self, tmp_path):
        bad = tmp_path / "source.rtf"
        bad.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError):
            StudioSession().load_script(bad)

    def test_generate_writes_script_and_manifest(self, source_file, tmp_path):
        session = StudioSession()
        session.load_script(source_file)
        output = session.generate_recording_script(tmp_path / "out", styles=["neutral"])
        assert output.markdown_path.exists()
        assert output.manifest_path.exists()
        assert session.last_output is output

    def test_generate_before_load_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            StudioSession().generate_recording_script(tmp_path)

    def test_record_is_a_placeholder(self, source_file):
        session = StudioSession()
        session.load_script(source_file)
        assert "not implemented" in session.record_line(0).lower()

    def test_record_without_selection_is_safe(self, source_file):
        session = StudioSession()
        session.load_script(source_file)
        assert "select" in session.record_line(-1).lower()


class TestIntegrationSeam:
    def test_events_reach_an_installed_sink(self, source_file):
        events = []
        integration_seam.set_event_sink(
            lambda app, event, payload: events.append((app, event))
        )
        try:
            StudioSession().load_script(source_file)
        finally:
            integration_seam.set_event_sink(None)
        assert ("Voice Studio", "script_loaded") in events

    def test_a_failing_sink_does_not_break_the_app(self, source_file):
        def boom(app, event, payload):
            raise RuntimeError("bus down")

        integration_seam.set_event_sink(boom)
        try:
            assert StudioSession().load_script(source_file).lines
        finally:
            integration_seam.set_event_sink(None)


class TestStudioWindow:
    def test_window_builds_and_lists_lines(self, studio_tk_root, source_file):
        from voiceclonegpt.studio_app.ui import StudioWindow

        window = StudioWindow(studio_tk_root)
        assert str(window.record_button["state"]) == "disabled"

        window.load_path(source_file)
        assert window.lines.size() == 2
        assert str(window.record_button["state"]) == "normal"
        assert "2 lines" in window.status_var.get()

    def test_load_error_shows_in_status_bar(self, studio_tk_root, tmp_path):
        from voiceclonegpt.studio_app.ui import StudioWindow

        window = StudioWindow(studio_tk_root)
        window.load_path(tmp_path / "missing.txt")
        assert "not found" in window.status_var.get().lower()
        assert window.lines.size() == 0

    def test_open_uses_the_file_picker_seam(self, studio_tk_root, source_file):
        from voiceclonegpt.studio_app.ui import StudioWindow

        window = StudioWindow(studio_tk_root, choose_file=lambda: str(source_file))
        window.on_open()
        assert window.lines.size() == 2

    def test_cancelled_picker_leaves_the_window_alone(self, studio_tk_root):
        from voiceclonegpt.studio_app.ui import StudioWindow

        window = StudioWindow(studio_tk_root, choose_file=lambda: "")
        window.on_open()
        assert window.lines.size() == 0
        assert window.source_var.get() == "No script loaded."

    def test_studio_does_not_import_the_reader(self):
        import voiceclonegpt.studio_app.core as core
        import voiceclonegpt.studio_app.ui as ui

        for module in (core, ui):
            source = open(module.__file__, encoding="utf-8").read()
            assert "reader_app" not in source


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

