# Cosmic Vine

A procedural 3D structure of connected cubes that grows outward like a vine.
Each block originates from the face of the previous one with a random
pitch/roll/yaw rotation, producing a wandering, non-axis-aligned chain where
every block is **perfectly flush** with its neighbour.

**Live demo →** https://samanseifi.github.io/cosmic-vine/

---

## The flush-face constraint

The naive approach — `T(b) @ R` — places the rotation pivot at the **centre**
of the next block.  This shifts the incoming face away from the outgoing face,
creating gaps.

The correct step decomposes the move as:

```
step = T(b/2) @ R @ T(b/2)
```

- `T(b/2)` — advance to the connecting face in the current local frame
- `R`       — rotate there (pivot is now at the shared face plane)
- `T(b/2)` — advance to the centre of the next block in its new local frame

This guarantees the −Z face of block N+1 lands exactly on the +Z face of
block N, to floating-point precision, for any rotation.

---

## Python usage

```bash
pip install -r requirements.txt
python cosmic_vine.py          # generates formx_cosmic_vine.obj + .png
```

```python
from cosmic_vine import CosmicVine, VineExporter

vine = CosmicVine(block_size=1.0)
vine.grow(iterations=100, max_rotation_rad=3.14159 / 4)
VineExporter.save(vine, 'vine.obj')
VineExporter.screenshot(vine, 'vine.png')
```

---

## Tests

```bash
pytest test_all.py -v
```

Key test: `test_flush_face_connectivity` — verifies the face-to-face
contact condition mathematically for all consecutive block pairs.

---

## Web app

`docs/index.html` is a self-contained Three.js app (no build step).
- **Blocks** slider (5 – 500)
- **Max Rotation** slider (0° – 90°)
- **Seed** slider for reproducible vines
- **Download .OBJ** exports the current vine

Deployed automatically to GitHub Pages via `.github/workflows/deploy.yml`
whenever a commit is pushed to `main`.

> **One-time setup**: in your repo go to
> *Settings → Pages → Build and deployment → Source → GitHub Actions*

---

## Project structure

```
├── cosmic_vine.py          # Core Python module
├── strand.py               # Strand / path variant
├── test_all.py             # Unit tests (pytest)
├── requirements.txt
├── docs/
│   └── index.html          # GitHub Pages web app (Three.js)
└── .github/
    └── workflows/
        └── deploy.yml      # CI: run tests → deploy Pages
```
