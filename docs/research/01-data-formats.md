# Research: Input data, formats, toolchain

Annotated, verified primary-source links for NRRD, pynrrd, 3D Slicer/MRHead, and Git LFS — for citing in the SRS/README instead of restating format details. All URLs fetched and resolved 2026-10-31.

## NRRD file format (official spec)

- [Definition of NRRD File Format — Teem (canonical spec)](https://teem.sourceforge.net/nrrd/format.html) — cite as the normative definition of the whole format so the SRS never re-describes headers, magic lines (`NRRD000X`), attached vs detached headers, or field grammar.
- [NRRD spec §4 "Space and Orientation Information"](https://teem.sourceforge.net/nrrd/format.html#space) — one line of cited text covers `space:`, `space origin:`, `space directions:`, and how the spec already defines the voxel-index ↔ world-space affine transform ("we did not use it" needs no explanation paragraph).
- [NRRD spec `space origin:` field](https://teem.sourceforge.net/nrrd/format.html#spaceorigin) — for the exact wording that the origin is the center of the first sample (same info as DICOM's Image Position tag).
- [NRRD spec `space directions:` field](https://teem.sourceforge.net/nrrd/format.html#spacedirections) — for the exact wording that each per-axis vector encodes direction *and* spacing (so anisotropic spacing is defined by the format, not invented here).
- [NRRD spec §6 per-axis `kinds:` field](https://teem.sourceforge.net/nrrd/format.html#kinds) — cite for the definition of axis `kinds` (e.g. the `domain domain domain` seen in MRHead_1.nrrd) instead of paraphrasing it.
- [NRRD spec `encoding:` field](https://teem.sourceforge.net/nrrd/format.html#encoding) — table of `raw`/`ascii`/`hex`/`gzip`/`bzip2` encodings with byte-level meaning; supports "input files use `encoding: gzip`" in one line.
- [NRRD spec §1.4 axis ordering + minimalist header](https://teem.sourceforge.net/nrrd/format.html#general.4) — cite for the spec's own definition of the 3D raster ("fastest to slowest" axis ordering, no assumed axis names) — this IS the project's raster model, so it can be quoted instead of defined.

## pynrrd (official docs / repo)

- [pynrrd official documentation (stable)](https://pynrrd.readthedocs.io/en/stable/) — the library reference to cite in the SRS dependencies; avoids explaining why a third-party reader implements the spec.
- [pynrrd examples: reading/writing files and headers](https://pynrrd.readthedocs.io/en/stable/examples.html) — official code showing `nrrd.read(filename)` returning `(data, header)` and header-only access; replace "how we load the volume" description with this link.
- [pynrrd `nrrd.read` reference](https://pynrrd.readthedocs.io/en/stable/reference/reading.html) — documents header fields exposure (`'space'`, `'space directions'`, typing) so the SRS can say "header fields accessed per this reference".
- [pynrrd "About the NRRD file format"](https://pynrrd.readthedocs.io/en/stable/background/about.html) — secondary background page linking back to the Teem spec; useful relay link in a README.
- [pynrrd on PyPI](https://pypi.org/project/pynrrd/) — pinned, versioned dependency (`pip install pynrrd`) for build/run instructions.
- [pynrrd source code (GitHub, mhe/pynrrd)](https://github.com/mhe/pynrrd) — cite the implementation repo itself for questions of behavior beyond the docs.

## 3D Slicer + MRHead (provenance of the MRI example)

- [3D Slicer docs: Coordinate systems (IJK vs LPS/RAS)](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html) — official page defining voxel-index (IJK) vs anatomical (RAS/LPS) coordinates, origin/spacing→world affine; one sentence covers "MRHead was inspected in Slicer using IJK coordinates".
- [Slicer SampleData source: MRHead donation statement](https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py) — primary provenance: "MRHead... donated to 3D Slicer project by the persons visible in the images, to be used without any restrictions"; cite instead of writing your own data-rights paragraph.
- [Slicer docs: Sample Data module](https://slicer.readthedocs.io/en/latest/user_guide/modules/sampledata.html) — official description of where Slicer users get MRHead, linking the repo asset to the standard workflow.
- [MRHead NRRD asset — Slicer/SlicerTestingData release (SHA256 content-addressed)](https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93) — the exact binary Slicer downloads as MR-head; a tamper-evident permalink (matches repo's MRHead_1.nrrd: 256×256×130, `type: short`, `space: left-posterior-superior`, `encoding: gzip`).

## Git LFS (large assets: .nrrd / .mp4)

- [Git LFS official site](https://git-lfs.com) — the homepage to link instead of explaining why big binaries don't belong in plain Git history.
- [Git LFS documentation (official repo /docs)](https://github.com/git-lfs/git-lfs/tree/main/docs) — primary user + API docs (`git lfs install/track/migrate`) for the README's one-line "assets are tracked with Git LFS".
- [Git LFS source repository (GitHub)](https://github.com/git-lfs/git-lfs) — the tool itself (releases, install instructions per-OS) for the "clone prerequisites" bullet.

## Unsigned integer types and NRRD `type:` values

- [NRRD spec `type:` field (full type table)](https://teem.sourceforge.net/nrrd/format.html#type) — the exact spec table mapping `"unsigned char"/"uint8"/"uint8_t"`, `"ushort"/"uint16"`, `"uint"/"uint32"`, `"short"/"int16"` → byte widths and C types; cite this instead of writing your own "8/16/32-bit unsigned" table (matches M being 8/16/32-bit, and MRHead_1.nrrd's `type: short` = **signed** int16 — the table says so verbatim).

## Caveats

- The canonical NRRD spec remains the Teem SourceForge page (last spec revision for Teem 1.9/NRRD0005). NRRD development now lives on GitHub, but there is no spec page there; SourceForge is the stable primary source, not a stale copy.
- `pynrrd`'s docs use a `/en/stable/` version slug pinned by the project (not Read the Docs' mutable `/latest/`), so the URLs above are safe to cite.
- Slicer docs: `https://www.slicer.org/wiki/...`-era coordinate pages were retired; the current official home is `slicer.readthedocs.io`. Older third-party tutorials still dominate search results for "Slicer coordinate systems" — avoid citing those.
- The MRHead provenance statement ("donated ... without any restrictions") lives inside a Python source file, not a standalone license page — cite the linked file/line.
- No dedicated official page exists just for "uint types in NRRD"; the spec's `type:` anchor is the best (and sufficient) primary source.
