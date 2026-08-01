# Verification — script generator (red/green record)

Evidence for the two hardening passes on `voiceclonemlx.recording`. Every test
below was written before the code that makes it pass, run to observe the
failure, and re-run after the change.

Command used throughout:

```bash
python3 -m pytest tests/voice_studio/test_script_generator.py
```

---

## Pass 1 — stable IDs, Markdown safety, weak-test repair

### Red

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py
5 failed, 25 passed
```

| Failing test | Why it failed |
|---|---|
| `TestStableIds::test_ids_are_content_derived` | IDs came from a per-run counter, so two entirely different sentences both received `NEUTRAL-001`. The ID encoded position, not content — the docstring's "stable" claim was false. |
| `TestStableIds::test_ids_survive_insertion_of_earlier_sentence` | Prepending one sentence shifted the counter for every later utterance, renumbering IDs that recordings may already be keyed to. |
| `TestSessionAndBlockBoundaries::test_markdown_includes_session_markers` | The Markdown had no session marker at all. The pre-existing assertion (`"Session" in markdown or "NEUTRAL" in markdown`) passed only through its `or` branch, so the missing marker went unnoticed. |
| `TestMarkdownSafety::test_source_text_markup_is_escaped` | Source text was interpolated raw, so `**bold**`, `` `code` ``, and `[link](http://evil)` rendered as live Markdown. |
| `TestMarkdownSafety::test_newlines_in_source_cannot_break_the_line` | A newline inside a sentence let source text emit `## Injected block` as a real heading, restructuring the script. |

### Green

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py
30 passed
$ python3 -m pytest
108 passed
```

Changes: `generate_stable_id` became a SHA-256 digest over style + source
filename + text; `_disambiguate` handles genuinely repeated text; the renderer
gained `escape_markdown` and an explicit `*Session {n}*` header. Three weak
tests were tightened (both sources required, literal session marker required,
coverage asserted as exact values).

---

## Pass 2 — renderer and store extracted into seams

### Red

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py -k "RendererSeam or StoreSeam"
7 failed, 30 deselected
```

| Failing test | Why it failed |
|---|---|
| `TestRendererSeam::test_render_script_is_importable_and_pure` | `ModuleNotFoundError: No module named 'voiceclonemlx.recording.script_rendering'` — rendering was a private function inside the coordinator, unreachable without running the whole pipeline against real files. |
| `TestRendererSeam::test_render_script_emits_one_section_per_style` | Same import failure; section structure could previously only be asserted through generated output. |
| `TestRendererSeam::test_render_script_skips_styles_without_utterances` | Same import failure. |
| `TestRendererSeam::test_escape_markdown_neutralizes_structure` | Same import failure; escaping had no direct unit test, only end-to-end assertions. |
| `TestStoreSeam::test_write_manifest_writes_one_json_object_per_line` | `ModuleNotFoundError: No module named 'voiceclonemlx.recording.script_storage'`. |
| `TestStoreSeam::test_write_manifest_overwrites_previous_content` | Same import failure. |
| `TestStoreSeam::test_write_manifest_preserves_unicode` | Same import failure. This test also pinned a real gap: the manifest writer used default `json.dumps`, which escapes non-ASCII to `\uXXXX`. |

### Green

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py
37 passed
$ python3 -m pytest
115 passed
```

Changes: `script_rendering.py` (`render_script`, `escape_markdown`) and
`script_storage.py` (`write_manifest`) extracted verbatim from the coordinator,
which now parses, constructs, sequences, and calls the two seams. Neither
module imports the coordinator — utterances are duck-typed — so both are
testable with hand-built objects and no file parsing. `write_manifest` gained
`ensure_ascii=False` and an explicit UTF-8 encoding to satisfy the Unicode test.

All prior fixes (content-derived IDs, collision suffixes, Markdown escaping,
explicit session marker, exact coverage assertions) remain covered and passing.

---

## Pass 3 — both outputs written through storage

Pass 2 left the manifest behind `script_storage` but the Markdown still went to
disk via `markdown_path.write_text(...)` inside the coordinator: one output had
a seam, the other did not.

### Red

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py -k "StoreSeam"
2 failed, 3 passed, 34 deselected
```

| Failing test | Why it failed |
|---|---|
| `TestStoreSeam::test_write_script_markdown_writes_the_file` | `ImportError` — no `write_script_markdown` existed; Markdown writing had no home in storage and no direct test. |
| `TestStoreSeam::test_both_outputs_are_written_through_storage` | `AttributeError: module 'script_generator' has no attribute 'write_script_markdown'` — with only the manifest seam stubbable, there was no way to assert the coordinator performs no writes of its own. |

### Green

```
$ python3 -m pytest tests/voice_studio/test_script_generator.py
39 passed
$ python3 -m pytest
117 passed
```

Changes: `script_storage.write_script_markdown(path, content)` added (explicit
UTF-8, replaces any existing file); `generate_script` calls it instead of
`Path.write_text`. `grep -n "write_text" script_generator.py` now returns
nothing — the coordinator performs no direct file writes. The new seam test
stubs both storage functions and asserts they are called in order with the
returned paths, and that with both stubbed **neither output file appears on
disk** — proving the coordinator has no write path of its own.

## Current test layout

The once-large test module is now split by responsibility:

- `tests/voice_studio/test_script_generation_core.py`
- `tests/voice_studio/test_script_generation_seams.py`

The historical commands above refer to the pre-split file; run both current
files together for the maintained suite.
