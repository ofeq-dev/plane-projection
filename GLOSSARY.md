# Glossary

Seed terms for the plane-projection domain. Refine and expand via `/domain-modeling` as decisions get made.

## Terms

- **Matrix (`M`)** — the 3D input array of voxels. Dimensions `sx, sy, sz` each satisfy `1 <= s < 2^10`. Stored as `.nrrd` files; values are unsigned 8-, 16-, or 32-bit integers.
- **Voxel (`v`)** — a single element of `M`, i.e. `v = M[x][y][z]`, always an unsigned integer.
- **Plane (`p`)** — a flat geometric target defined by 3 or 4 coordinates. A 3-point plane satisfies `|dist(a,b) - dist(a,c)| < 0.001`; a 4-point plane satisfies the three validity rules in `AGENTS.md`. An invalid plane raises an exception.
- **Plane set** — 1+ planes from a `.csv` input file, combined for one output video.
- **Projection** — the mapping of `M`'s voxel values onto a plane `p`; the per-plane result is a 2D frame of voxel values.
- **Volume** — the loaded 3D matrix M as an input file (or the array it parses to); the NRRD-input synonym for Matrix.
- **Closest voxel** — the selection rule: when computing a projection, choose the nearest voxel rather than a more complex sampling solution.
- **Frame** — one rendered plane projection; frames are combined into an output video (`.mp4`).
- **Error frame** — a black video frame with a red banner in a fixed corner giving the CSV row index and the specific reason a plane could not produce a frame (failed validity rule or degenerate geometry). The video continues with the remaining planes; the same line is logged to the CLI.
- **Canonical space** — voxel-index space: all plane coordinates and lookups live in `M[x][y][z]` array coordinates. World-space header fields (`space origin`, `space directions`) are deliberately unused; see ADR 0002 (to be written).
- **Frame** *(refined)* — a 2D image one sample per voxel-length unit: height = leg length in index units (rounded), width = base length (rounded). Apex `a` at top-center, legs running downward to b/c, rows centered on the canvas; no mirroring.
- **Canvas** — the fixed video frame size: the maximum frame width and height across all planes in one CSV, all frames padded top-left with black to fit, since one video requires one fixed frame size.
- **Sample rule** — closest voxel by half-to-even rounding of the sample point's index coordinates.
