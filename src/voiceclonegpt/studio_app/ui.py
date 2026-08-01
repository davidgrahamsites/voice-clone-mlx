"""Tkinter window for Voice Studio.

The window owns no logic: it drives a :class:`StudioSession` and renders the
result. Constructing ``StudioWindow`` never enters an event loop, so tests can
build it against a withdrawn root.
"""

import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

from . import APP_NAME
from .core import StudioSession


class StudioWindow:
    """Load a script file, review its lines, and (later) record them.

    ``choose_file`` is the seam for the native open dialog: tests pass a stub
    so the file-picking path can run without a live dialog.
    """

    def __init__(
        self,
        master: tk.Misc,
        session: StudioSession = None,
        choose_file=None,
    ) -> None:
        self.master = master
        self.session = session or StudioSession()
        self.choose_file = choose_file or self._ask_open_filename
        master.title(APP_NAME)

        frame = ttk.Frame(master, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X)
        self.open_button = ttk.Button(
            toolbar, text="Open Script…", command=self.on_open
        )
        self.open_button.pack(side=tk.LEFT)
        self.record_button = ttk.Button(
            toolbar, text="Record Selected", command=self.on_record, state=tk.DISABLED
        )
        self.record_button.pack(side=tk.LEFT, padx=(8, 0))

        self.source_var = tk.StringVar(value="No script loaded.")
        ttk.Label(frame, textvariable=self.source_var).pack(
            fill=tk.X, pady=(10, 4), anchor=tk.W
        )

        self.lines = tk.Listbox(frame, height=16, activestyle="dotbox")
        self.lines.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Open a .txt script to begin.")
        ttk.Label(frame, textvariable=self.status_var).pack(
            fill=tk.X, pady=(8, 0), anchor=tk.W
        )

    @staticmethod
    def _ask_open_filename() -> str:
        return filedialog.askopenfilename(
            title="Choose a script file", filetypes=[("Text files", "*.txt")]
        )

    def on_open(self) -> None:
        path = self.choose_file()
        if path:
            self.load_path(path)

    def load_path(self, path) -> None:
        """Load a source file and show its lines. Errors land in the status bar."""
        try:
            script = self.session.load_script(path)
        except (FileNotFoundError, ValueError, NotImplementedError) as exc:
            self.status_var.set(str(exc))
            return

        self.lines.delete(0, tk.END)
        for index, line in enumerate(script.lines, start=1):
            self.lines.insert(tk.END, f"{index:03d}  {line}")
        self.source_var.set(str(Path(path)))
        self.record_button.configure(
            state=tk.NORMAL if script.lines else tk.DISABLED
        )
        self.status_var.set(f"{len(script.lines)} lines ready.")

    def on_record(self) -> None:
        selection = self.lines.curselection()
        index = selection[0] if selection else -1
        self.status_var.set(self.session.record_line(index))


def build_app() -> tk.Tk:
    """Create the root window and its Studio UI without starting the event loop."""
    root = tk.Tk()
    StudioWindow(root)
    return root


def main() -> None:
    build_app().mainloop()

