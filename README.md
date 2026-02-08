# fingerprint-research (Unit 1)

Reproducible fingerprint matching experiments + a small demo API/UI.

Implemented matchers:
- **classic_v2**: keypoints + matching + geometry (ORB/GFTT + RANSAC)
- **dl_quick**: lightweight deep embedding baseline
- **dedicated**: learned patch-descriptor matcher (trained Week 6–7, evaluated Week 8+)

## Repo layout
- `src/` – core preprocessing + matching code
- `scripts/` – week-by-week scripts + automation (`scripts/run_all.py`)
- `api/` – FastAPI service (`/health`, `/match`)
- `ui/` – Vite + React demo client
- `tests/` – sanity tests

> This repo does **not** include fingerprint datasets. You must obtain SD300B (or another dataset) separately and generate processed files locally.

## Setup (Python)
```bash
conda env create -f environment.yml
conda activate fingerprint_research
