# ADR 0004 — Video encoding via OpenCV's `cv2.VideoWriter` (`mp4v`)

- **Status:** Accepted — supersedes [ADR 0003](0003-video-encoding-via-ffmpeg-cli.md)
- **Date:** 2026-11-02
- **Deciders:** ofeq-dev (PR #7 review)

## Context

ADR 0003 chose the system `ffmpeg` CLI over OpenCV's `cv2.VideoWriter` to satisfy
FR-19's H.264 codec requirement, because pip-installable OpenCV builds of the time
could not open an H.264 fourcc. That decision made `ffmpeg` a build/run prerequisite
on PATH, and pip-installable `opencv-python` was still a rejected alternative.

## Decision drivers

- PR #7 review feedback: "Let's switch to `mp4v` - it's probably nicer to keep
  everything under pip (thus `ffmpeg` no longer needed on PATH)."
- Keeping the dependency triad (Python + numpy + pynrrd, NFR-3) pip-only is worth
  more than the specific codec: a hard system binary requirement hurts
  reproducibility (CI, graders' machines) for a tool whose mp4 output is a
  delivery container, not a distribution asset.
- `mp4v` (MPEG-4 Part 2) is universally decodable in the contexts this tool
  targets (browsers, media players, notebooks), even though it is not H.264.

## Decision

Encode with `cv2.VideoWriter` using the `mp4v` fourcc into an mp4 container, at
30 FPS, via `opencv-python` (pip). Drop the system `ffmpeg` requirement entirely.

Consequences:

- FR-19's codec clause changes from "H.264-compatible fourcc" to "MPEG-4 Part 2
  (`mp4v`) fourcc"; the even-dimension requirement is kept (OpenCV's underlying
  encoders share yuv420p-style chroma subsampling constraints, and the CLI
  contract already validated it).
- `requirements.txt` gains `opencv-python`; no PATH prerequisite is documented.
- A future ticket may revisit H.264 (e.g. via `imageio-ffmpeg` or a hardware
  backend) if the deliverable ever demands it; that would be a new ADR.

## Alternatives considered

- **Keep ADR 0003 (`ffmpeg` CLI, H.264)**: rejected — external binary on PATH is
  a reproducibility and setup burden; review feedback preferred pip-only deps.
- **`cv2.VideoWriter` with `avc1` (H.264)**: rejected on prior evidence (never
  opens with the pip OpenCV builds tried on this machine).
- **`imageio-ffmpeg`** (bundled static ffmpeg, ~60 MiB wheels): rejected — heavy,
  and keeps a second ffmpeg copy per environment; overkill for placeholder frames.
