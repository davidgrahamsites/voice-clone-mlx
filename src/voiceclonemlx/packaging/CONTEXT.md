---
type: module-contract
module: macos-app-packaging
sequence: semantic
---

# packaging — macOS .app bundles

One job: write a minimal, deterministic `.app` directory that launches one of
the two apps. It is a *wrapper*, not a distribution: it bundles no Python
runtime, downloads nothing, signs nothing.

## Inputs

- An app id from `APP_SPECS` — `voice_studio` or `voice_reader`. The registry
  is the allowlist; a caller cannot name an arbitrary module or bundle name.
- An **existing** output directory.
- A repo root containing `src/`.
- Optional version string and `replace` flag.

Do NOT load: app state, model bundles, ingestion code, or the network.

## Process

1. Resolve the app id against the registry; anything else raises
   `UnknownAppError` naming the known ids.
2. Resolve and check the output directory and repo root. A missing directory,
   a file where a directory belongs, or a repo without `src/` raises
   `UnsafeOutputError` — the `src/` check exists because a bundle pointing at a
   repo without it would fail only at launch, far from the cause.
3. If the bundle already exists, refuse with `BundleExistsError` unless
   `replace=True`.
4. Write `Contents/Info.plist` — including the `VCMLXBuiltBy` marker — and
   `Contents/MacOS/<Bundle Name>`, then mark the launcher executable.

## Outputs

```text
<output>/Voice Studio.app/
  Contents/
    Info.plist
    MacOS/Voice Studio       # executable shell launcher
```

The launcher derives the repo from its **own** location:

```sh
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here"/../../../.. && pwd)
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m voiceclonemlx.studio_app
```

The hop is a separate, shell-quoted token — see the quoting rule below.

## Determinism, and its exact limit

`Info.plist` contains no timestamp and no path, so it is byte-identical
wherever and whenever it is built — asserted by
`test_plist_is_identical_wherever_it_is_built` and by a test that no `date`
key exists.

The launcher is **not** location-independent: it embeds the relative hop from
the bundle to the repo, so building into a different depth produces different
bytes. That is the deliberate trade-off for baking in no absolute path — and
it is asserted rather than left implicit
(`test_launcher_hop_reflects_the_output_location`). Consequence for users:
**moving a bundle on its own breaks it**; move it with the repo, or rebuild.

## Safety rules

### The launcher is generated, so the hop is shell-quoted

`relative_repo` comes from a filesystem path, which is untrusted text. It is
written through `shlex.quote`, as a separate token:

```sh
repo=$(cd "$here"/'../../../../evil$(touch /tmp/PWNED)' && pwd)
```

Interpolating it into a double-quoted string instead — which an earlier version
did — makes a repo directory named with `` ` ``, `$( )`, or `"` execute
arbitrary commands **every time the app is launched**. Five hostile names are
parametrized across three tests each: exact-line equality with the quoted
template, a real `/bin/sh` run asserting no injected command fires, and a
positive test that quoting has not broken the ordinary case.

If you edit `build_launcher`, keep every path-derived value quoted.

### Nothing is deleted implicitly

`replace=True` is required to rebuild. Even then `_remove_bundle` refuses
unless **all** of these hold:

| Check | Why |
|---|---|
| Path resolves inside the output directory | no deletion outside where the caller pointed |
| Path is a directory | not a file or dangling link |
| `Info.plist` parses as a dict | unreadable plist proves nothing |
| `VCMLXBuiltBy == "voiceclonemlx.packaging.app_bundle"` | **an `Info.plist` alone proves nothing — every macOS app has one.** Only a bundle carrying this tool's marker may be removed |
| `CFBundleIdentifier == spec.bundle_id` | the bundle belongs to *this* app. Without it, rebuilding Studio would delete a Reader bundle that had been renamed `Voice Studio.app` |

The marker is written into every bundle we build (`MARKER_KEY` / `MARKER`), so
it is also a constant — it adds no build-time variability and determinism is
unaffected.

Removal walks bottom-up with `os.walk`/`unlink`/`rmdir`; there is no recursive
tree delete and `shutil` is deliberately not imported.

### No subprocess, no network, no git

Tests assert the module imports none of `urllib`, `requests`, `socket`,
`subprocess`, or `shutil`, and that the source contains no `os.system`,
`Popen`, `check_call`, or `git`. `shlex` is stdlib and text-only.

The repo is never written to; only the output directory is touched.

## Human check

1. Build into a scratch directory first:
   `python3 scripts/build_apps.py --output /tmp/vcg-apps`
2. Double-click `Voice Studio.app` in Finder. It should open the Studio window.
   Gatekeeper will warn on first launch — the bundle is unsigned and
   un-notarized by design; right-click → Open to approve it once.
3. Confirm the app still launches after the repo moves *with* the bundle, and
   understand it will not after the bundle moves alone.
4. If `--replace` refuses, read the message before forcing anything. "not built
   by this tool" or "belongs to a different app" means something else owns that
   name — inspect it by hand. Do not delete it to make the build pass.
5. Before shipping to anyone else, revisit this contract: an unsigned wrapper
   around a system `python3` is right for a personal local utility and wrong
   for distribution.

## Not implemented

Icons, a bundled Python runtime, codesigning, notarization, DMG packaging,
universal binaries. Each is a separate decision; none is implied by this
module.

## Tests

`tests/packaging/test_app_bundle.py` — 62 tests: registry and unknown-id
refusal, bundle layout, plist contents (including the marker), launcher
behaviour, **shell-quoting against five hostile repo names**, determinism and
its limit, the **marker and identity checks** before any deletion, output
confinement, and the no-network/no-subprocess checks. Red/green evidence:
`docs/verification/app-packaging-tdd.md`.

