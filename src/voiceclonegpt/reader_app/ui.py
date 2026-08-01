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
    ) -> None:
        self.master = master
        self.player = player or Player()
        self.choose_file = choose_file or self._ask_open_filename
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
    root = tk.Tk()
    ReaderWindow(root)
    return root


def main() -> None:
    build_app().mainloop()

