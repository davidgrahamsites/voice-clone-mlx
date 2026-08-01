---
type: module-contract
module: developer-scripts
sequence: semantic
---

# scripts — bounded developer entry points

Scripts in this folder are thin command-line adapters. They parse arguments and
delegate all application rules to a source module. They must not download,
install, call external services, or perform destructive cleanup.

The current entry point is build_apps.py, which delegates deterministic
macOS-wrapper creation to voiceclonemlx.packaging.app_bundle.
