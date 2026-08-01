---
type: module-contract
module: packaging-tests
sequence: semantic
---

# tests/packaging — packaging contract tests

These tests build only temporary .app directory trees. They verify the
allowlisted app ids, plist metadata, executable launcher, path derivation, and
deterministic rebuild behavior without launching a GUI or touching the repo.

- `test_app_bundle.py` covers registry, layout, plist, launcher content,
  determinism, and boundedness.
- `test_app_bundle_security.py` covers shell quoting, output confinement,
  replacement, and destructive-safety rules.
