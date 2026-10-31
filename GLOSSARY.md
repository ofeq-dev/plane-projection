# Glossary

Seed terms for the plane-projection domain. Refine and expand via `/domain-modeling` as decisions get made.

## Terms

- **Matrix (`M`)** — the 3D input array of voxels. Dimensions `sx, sy, sz` each satisfy `1 <= s < 2^10`. Stored as `.nrrd` files; values are unsigned 8-, 16-, or 32-bit integers.
- **Voxel (`v`)** — a single element of `M`, i.e. `v = M[x][y][z]`, always an unsigned integer.
- **Plane (`p`)** — a flat geometric target defined by 3 or 4 coordinates. A 3-point plane satisfies `|dist(a,b) - dist(a,c)| < 0.001`; a 4-point plane satisfies the three validity rules in `AGENTS.md`. An invalid plane raises an exception.
- **Plane set** — 1+ planes from a `.csv` input file, combined for one output video.
- **Projection** — the mapping of `M`'s voxel values onto a plane `p`; the per-plane result is a 2D frame of voxel values.
- **Closest voxel** — the selection rule: when computing a projection, choose the nearest voxel rather than a more complex sampling solution.
- **Frame** — one rendered plane projection; frames are combined into an output video (`.mp4`).
- **Error frame** — a special video frame conveying that a plane was invalid and the algorithm raised an exception.
