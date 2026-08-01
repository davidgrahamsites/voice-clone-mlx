"""Tkinter window for Voice Reader.

Constructing ``ReaderWindow`` never enters an event loop, so tests can build it
against a withdrawn root.
"""

import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

from . import APP_NAME
from .core import Player, load_manifest


class ReaderWindow:
    """Open a recording manifest, list its takes, and play the recorded ones.

    ``choose_file`` is the seam for the native open dialog: tests pass a stub
    so the file-picking path can run without a live dialog.
    """

    def __init__(
        self,
        master: tk.Misc,
        player: Player = None,
        choose_file=None,
        synthesis_controller=None,
        choose_bundle=None,
        choose_text_file=None,
        choose_output_dir=None,
    ) -> None:
        self.master = master
        self.player = player or Player()
        self.choose_file = choose_file or self._ask_open_filename
        self.synthesis_controller = synthesis_controller
        self.choose_bundle = choose_bundle or self._ask_bundle_directory
        self.choose_text_file = choose_text_file or self._ask_text_filename
        self.choose_output_dir = choose_output_dir or self._ask_output_directory
        self.bundle_dir = None
        self._voice_ready = False
        self.takes = []
        master.title(APP_NAME)

        frame = ttk.Frame(master, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open Manifest…", command=self.on_open).pack(
            side=tk.LEFT
        )
        self.play_button = ttk.Button(
            toolbar, text="Play", command=self.on_play, state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(
            toolbar, text="Stop", command=self.on_stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.manifest_var = tk.StringVar(value="No manifest loaded.")
        ttk.Label(frame, textvariable=self.manifest_var).pack(
            fill=tk.X, pady=(10, 4), anchor=tk.W
        )

        self.take_list = tk.Listbox(frame, height=16, activestyle="dotbox")
        self.take_list.pack(fill=tk.BOTH, expand=True)
        self.take_list.bind("<Double-Button-1>", lambda _event: self.on_play())

        synthesis = ttk.LabelFrame(frame, text="Cloned voice synthesis", padding=8)
        synthesis.pack(fill=tk.X, pady=(10, 0))

        voice_row = ttk.Frame(synthesis)
        voice_row.pack(fill=tk.X)
        self.voice_button = ttk.Button(
            voice_row, text="Choose Voice Bundle…", command=self.on_choose_bundle
        )
        self.voice_button.pack(side=tk.LEFT)
        self.voice_var = tk.StringVar(value="No verified voice selected.")
        ttk.Label(voice_row, textvariable=self.voice_var).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        runtime_row = ttk.Frame(synthesis)
        runtime_row.pack(fill=tk.X, pady=(6, 0))
        self.runtime_var = tk.StringVar()
        self.runtime_box = ttk.Combobox(
            runtime_row, textvariable=self.runtime_var, state="readonly", width=24
        )
        self.runtime_box.pack(side=tk.LEFT)
        self.runtime_box.bind("<<ComboboxSelected>>", self._invalidate_voice)
        self.check_voice_button = ttk.Button(
            runtime_row, text="Check Voice", command=self.on_check_voice
        )
        self.check_voice_button.pack(side=tk.LEFT, padx=(8, 0))

        text_row = ttk.Frame(synthesis)
        text_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(text_row, text="Open Text…", command=self.on_open_text).pack(
            side=tk.LEFT
        )
        self.synthesis_text = tk.Text(synthesis, height=5, wrap=tk.WORD)
        self.synthesis_text.pack(fill=tk.X, pady=(6, 0))
        self.synthesis_text.bind("<KeyRelease>", self._sync_generate_state)

        generate_row = ttk.Frame(synthesis)
        generate_row.pack(fill=tk.X, pady=(6, 0))
        self.output_name_var = tk.StringVar(value="reader-output.wav")
        ttk.Entry(generate_row, textvariable=self.output_name_var, width=32).pack(
            side=tk.LEFT
        )
        self.generate_button = ttk.Button(
            generate_row,
            text="Generate WAV or MP3…",
            command=self.on_generate,
            state=tk.DISABLED,
        )
        self.generate_button.pack(side=tk.LEFT, padx=(8, 0))

        if self.synthesis_controller is None:
            for widget in (
                self.voice_button,
                self.runtime_box,
                self.check_voice_button,
                self.synthesis_text,
            ):
                widget.configure(state=tk.DISABLED)

        self.status_var = tk.StringVar(
            value="Open a script_session_N.jsonl manifest to begin."
        )
        ttk.Label(frame, textvariable=self.status_var).pack(
            fill=tk.X, pady=(8, 0), anchor=tk.W
        )

    @staticmethod
    def _ask_open_filename() -> str:
        return filedialog.askopenfilename(
            title="Choose a recording manifest", filetypes=[("Manifests", "*.jsonl")]
        )

    @staticmethod
    def _ask_bundle_directory() -> str:
        return filedialog.askdirectory(title="Choose a verified voice bundle")

    @staticmethod
    def _ask_text_filename() -> str:
        return filedialog.askopenfilename(
            title="Choose text", filetypes=[("Text files", "*.txt"), ("All files", "*")]
        )

    @staticmethod
    def _ask_output_directory() -> str:
        return filedialog.askdirectory(title="Choose WAV or MP3 output folder")

    def on_choose_bundle(self) -> None:
        path = self.choose_bundle()
        if not path:
            return
        self._voice_ready = False
        try:
            choice = self.synthesis_controller.inspect_bundle(path)
        except Exception as exc:
            self.bundle_dir = None
            self.runtime_box.configure(values=())
            self.runtime_var.set("")
            self.status_var.set(str(exc))
            self._sync_generate_state()
            return

        self.bundle_dir = Path(path)
        self.voice_var.set(choice.bundle_id)
        self.runtime_box.configure(values=choice.runtime_ids)
        self.runtime_var.set("")
        self.status_var.set("Choose a declared runtime, then check readiness.")
        self._sync_generate_state()

    def on_check_voice(self) -> None:
        self._voice_ready = False
        runtime_id = self.runtime_var.get()
        if self.bundle_dir is None or not runtime_id:
            self.status_var.set("Choose a bundle and one of its declared runtimes.")
            return
        try:
            report = self.synthesis_controller.select(self.bundle_dir, runtime_id)
        except Exception as exc:
            self.status_var.set(str(exc))
            self._sync_generate_state()
            return

        if report.ready:
            self._voice_ready = True
            self.status_var.set("Voice is verified and ready for local synthesis.")
        else:
            blockers = ", ".join(report.blockers) or "verification incomplete"
            self.status_var.set(f"Voice is not ready: {blockers}.")
        self._sync_generate_state()

    def on_open_text(self) -> None:
        path = self.choose_text_file()
        if not path:
            return
        try:
            text = self.synthesis_controller.ingest_text(path)
        except Exception as exc:
            self.status_var.set(str(exc))
            return
        self.synthesis_text.delete("1.0", tk.END)
        self.synthesis_text.insert("1.0", text)
        self.status_var.set(f"Loaded text from {Path(path)}.")
        self._sync_generate_state()

    def _sync_generate_state(self, _event=None) -> None:
        has_text = bool(self.synthesis_text.get("1.0", "end-1c").strip())
        ready = bool(
            self.synthesis_controller is not None
            and self._voice_ready
            and self.synthesis_controller.ready
            and has_text
        )
        self.generate_button.configure(state=tk.NORMAL if ready else tk.DISABLED)

    def _invalidate_voice(self, _event=None) -> None:
        """Require readiness again after the displayed runtime changes."""
        self._voice_ready = False
        self._sync_generate_state()

    def on_generate(self) -> None:
        output_dir = self.choose_output_dir()
        if not output_dir:
            return
        try:
            self.synthesis_controller.set_text(
                self.synthesis_text.get("1.0", "end-1c")
            )
            output = self.synthesis_controller.synthesize(
                output_dir, self.output_name_var.get()
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            self._sync_generate_state()
            return
        self.status_var.set(f"Generated {output}.")

    def on_open(self) -> None:
        path = self.choose_file()
        if path:
            self.load_path(path)

    def load_path(self, manifest_path, audio_dir=None) -> None:
        """Load a manifest and list its takes. Errors land in the status bar."""
        try:
            self.takes = load_manifest(manifest_path, audio_dir)
        except (FileNotFoundError, ValueError) as exc:
            self.status_var.set(str(exc))
            return

        self.take_list.delete(0, tk.END)
        for take in self.takes:
            self.take_list.insert(tk.END, take.label())
        self.manifest_var.set(str(Path(manifest_path)))
        state = tk.NORMAL if self.takes else tk.DISABLED
        self.play_button.configure(state=state)
        self.stop_button.configure(state=state)
        playable = sum(1 for take in self.takes if take.has_audio)
        self.status_var.set(f"{len(self.takes)} takes, {playable} with audio.")

    def on_play(self) -> None:
        selection = self.take_list.curselection()
        if not selection:
            self.status_var.set("Select a take to play.")
            return
        self.status_var.set(self.player.play(self.takes[selection[0]]))

    def on_stop(self) -> None:
        self.player.stop()
        self.status_var.set("Stopped.")


def build_app() -> tk.Tk:
    """Create the root window and its Reader UI without starting the event loop."""
    from .composition import build_synthesis_controller

    root = tk.Tk()
    ReaderWindow(root, synthesis_controller=build_synthesis_controller())
    return root


def main() -> None:
    build_app().mainloop()
