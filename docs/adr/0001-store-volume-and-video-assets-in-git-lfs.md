# ADR-0001: Store volumetric and video assets in Git LFS

- **Date**: 2026-10-31
- **Status**: Accepted

## Context

Example matrices in `assets/nrrd/` total ~95 MB (`croissant.nrrd` is 63 MB), and the project's output is `.mp4` videos. GitHub stores real git objects up to 100 MB and warns above 50 MB, so committing these files to plain git history would bloat every clone and risk rejected pushes.

## Decision

Track `*.nrrd` and `*.mp4` in [Git LFS](https://git-lfs.github.com/) via `.gitattributes`, before their first commit. Small assets (`.csv`, `.png`) stay in plain git.

## Consequences

- Large assets never enter normal git history; clones fetch LFS pointers and resolve them on demand.
- Anyone cloning needs `git lfs install` before pulling (already initialized locally).
- Future output videos must also be committed (not only left on disk) to be reproducible from the repo.
