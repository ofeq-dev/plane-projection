# Guidance to the LLM

## General

1. Include an AI disclosure in each PR.
2. Merges are only done by human to ensure _every single_ line is reviewed / explained / owned.
3. Commit frequently

## Project

**Goals**:

- Going from "not well defined" to "robustly designed"
- Writing documents like SOW, SRS, etc.
- Time efficiency + frequent commits
- Comprehensive testing suite

**Description**:

- Input - a 3D image `M` and 1+ 2D planes `p`
- Algorithm - calculate the values of the `M` for each projection `p`
- Output - a video combining each of these frames

**Details**:

- Each voxel `v = M[x][y][z]` is an unsigned integer
- A given `M` can be 8, 16, or 32 bit values
- Each dimension `sx,sy,sz` of `M` must be `1 <= s < 2^10`
- Each "plane" consists of 3 to 4 coordinates
  - 3 points fulfil `|dist(a,b) - dist(a,c)| < 0.001`
  - 4 points fulfil:
    1. `|dist(a1,b) - dist(a2,c)| < 0.001`
    2. `dist(a1,a2) <= dist(b,c)`
    3. a1,b and a2,c are on the same plane
- An invalid plane should raise an exception and the output video should convey this with an "error frame"
- Use python, while ensuring a fast runtime (at least 30 FPS for matrix of `500^3` bytes, though ideally over 100 FPS)
- Example matrices (each a `.nrrd` file) and planes (multiple in each `.csv` file) are provided
- The project should run via `python Ex0.py file.nrrd file.csv file.mp4`
- The algorithm should select the "closest voxel" (rather than a more complicated solution)

**TODO**:

1. [ ] Write an SRS
2. [ ] Write an algorithm
3. [ ] Define a testing strategy
4. [ ] Export an example run and include a write up
5. [ ] Ensure the repo contains 1) docs, 2) source code, 3) README (with build / run instruction), 4) benchmarks, videos, etc.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`ofeq-dev/plane-projection`) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
