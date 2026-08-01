#!/usr/bin/env python3
"""Build the two macOS .app bundles.

    python3 scripts/build_apps.py                 # into apps/
    python3 scripts/build_apps.py --replace       # rebuild over existing ones
    python3 scripts/build_apps.py --app voice_studio --output /tmp/out

Thin by design: argument parsing and printing only. Every rule about what a
bundle contains lives in `voiceclonegpt.packaging.app_bundle`.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from voiceclonegpt.packaging.app_bundle import (  # noqa: E402
    APP_SPECS,
    DEFAULT_VERSION,
    PackagingError,
    build_app_bundle,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--app",
        action="append",
        choices=sorted(APP_SPECS),
        help="app to build; repeatable. Default: all.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "apps"),
        help="directory to write bundles into (must exist)",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="rebuild over an existing bundle instead of refusing",
    )
    args = parser.parse_args(argv)

    for app_id in args.app or sorted(APP_SPECS):
        try:
            bundle = build_app_bundle(
                app_id,
                args.output,
                REPO_ROOT,
                version=args.version,
                replace=args.replace,
            )
        except PackagingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"built {bundle}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

