# Breaking migration: VoiceCloneGPT → VoiceCloneMLX

**Change level:** MAJOR (`1.0.0` migration)

VoiceCloneMLX is now the canonical product and Python package name. The rename
is intentional: the project is moving from a generic working title to the
MLX-first identity used by the local runtime and repository.

## What changed

- `src/voiceclonegpt/` became `src/voiceclonemlx/`.
- Imports changed from `voiceclonegpt.*` to `voiceclonemlx.*`.
- Module entry points changed to:

  ```bash
  PYTHONPATH=src python3 -m voiceclonemlx.studio_app
  PYTHONPATH=src python3 -m voiceclonemlx.reader_app
  ```

- App bundle identifiers changed to `com.voiceclonemlx.studio` and
  `com.voiceclonemlx.reader`.
- The generated bundle marker changed to
  `VCMLXBuiltBy = voiceclonemlx.packaging.app_bundle`.
- The UI-test environment variable changed from
  `VOICECLONEGPT_RUN_UI_TESTS` to `VOICECLONEMLX_RUN_UI_TESTS`.
- The public documentation, rendered pages, user-agent string, temporary
  prefixes, and visual assets use **VoiceCloneMLX**.

## Migration steps

1. Replace every `voiceclonegpt` import with `voiceclonemlx`.
2. Update scripts that set `PYTHONPATH` or invoke a module.
3. Rebuild both `.app` launchers; old bundles are not interchangeable because
   their marker and identifiers belong to the previous package.
4. Rename any stored test command using `VOICECLONEGPT_RUN_UI_TESTS`.
5. Do not edit old published model bundles in place. Their model identity and
   artifact checksums remain valid; only the application loader path changes.

The on-disk workspace folder was subsequently renamed to `VoiceCloneMLX`.
Update any saved shell shortcuts or open Orca workspaces that still point at
the former `VoiceCloneGPT` path.

## Compatibility decision

There is no silent compatibility alias. Keeping an old import package would
make it unclear which app identity created a model bundle and would allow old
launchers to appear valid after the rename. A caller that needs the old line
must stay on the pre-migration branch and migrate deliberately.
