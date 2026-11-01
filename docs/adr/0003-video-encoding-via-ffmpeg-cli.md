# ADR-0003: H.264 encoding via the system ffmpeg CLI

- **Date**: 2026-11-02
- **Status**: Proposed (pending owner review at merge)

## Context

The SRS (FR-18/FR-19) requires the output container to be mp4 with a genuine
H.264-compatible fourcc at a fixed 30 FPS container rate, uniform even frame
dimensions. FR-18's same-size constraint cites OpenCV's
[`cv::VideoWriter.write()`](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html)
as its origin, and the SRS's dependency triad (NFR-3) is Python + numpy +
pynrrd.

Empirically on this project's dev environment (pip `opencv-python` 5.0.0.93):

- `cv2.VideoWriter` with fourcc `avc1`/`H264` never opens (`isOpened() ==
  False`): the wheel's bundled ffmpeg build reaches only a `h264_v4l2m2m`
  hardware encoder ("Could not find a valid device ... Failed to initialize
  VideoWriter") — no libx264/openh264 software H.264 encoder is compiled in.
- The only working fourcc is `mp4v` (MPEG-4 Part 2), which is *not*
  H.264-compatible and therefore does not satisfy FR-19 literally.
- The system `ffmpeg` CLI (same machine) encodes H.264 through `libx264`.

## Decision

Write mp4 video by piping raw frames to the system `ffmpeg` CLI:

```
ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 \
       -s <W>x<H> -r 30 -i pipe:0 -an -c:v libx264 -pix_fmt yuv420p \
       -movflags +faststart <out.mp4>
```

- Video writer owns the canvas rule: even (W, H) enforced before encoding —
  odd canvas would be silently cropped by the CLI, quietly breaking FR-18's
  one-canvas semantics.
- Uniform frame size is enforced by construction: the writer is opened for
  one canvas and receives only frames of exactly that size (FR-18).
- The CLI-seam tests decode the produced file with `ffprobe` (`h264` codec,
  `mp4` container, frame count, fps) rather than re-rendering it.

## Alternatives considered

- **`cv2.VideoWriter` with `mp4v`**: simple and pip-installable everywhere,
  but encodes MPEG-4 Part 2 — fails FR-19's H.264 requirement outright.
- **`cv2.VideoWriter` with `avc1`**: rejected on the evidence above (never
  opens with any pip OpenCV build tried on this machine).

## Consequences

- `ffmpeg` is a build/run dependency of the video writer (single system
  binary; the render path remains numpy/pynrrd per NFR-3). README's build
  instructions must list it (repo To-Do #5).
- decoding/playback in tests similarly uses the ffmpeg toolchain, not a pip
  dependency.
- SRS FR-18's citation should mention that the same-size constraint applies
  to the ffmpeg CLI writer as well (equal restriction, different tool) — owner
  review updates the wording.
- A zero-frame run (CSV with no data rows) still writes the empty mp4
  container (ffmpeg exits 0); a decision point for the plane-list ticket
  whether a 0-row CSV should be rejected as usage (FR-2) instead.
