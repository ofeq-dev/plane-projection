# Canonical voxel-index space

NRRD files carry a world-space frame (`space origin`, `space directions`, with per-axis
spacing), which the [NRRD spec](https://teem.sourceforge.net/nrrd/format.html#space)
defines as an affine transform between voxel-index and physical coordinates. Testing
both interpretations of the provided CSV plane files against the provided volumes showed
the CSVs are written in voxel-index coordinates: reading them as index coordinates puts
64–96% of plane samples in-bounds, as world coordinates only 0–6%. The exercise brief
likewise defines the volume purely as `M[x][y][z]` with no physical units.

**Decision**: voxel-index space is the one canonical coordinate space. Plane coordinates
are used directly as array indices; `space origin` / `space directions` are not applied,
and anisotropic spacing is deliberately ignored.

**Consequences**: on volumes with non-uniform spacing (e.g. `onion` at 0.55 vs 0.715
mm), a geometrically square plane renders with a correspondingly stretched aspect —
accepted as "close enough" per the lecturer's guidance, traded against avoiding a
second coordinate system in the algorithm, tests, and cross-student comparison.
