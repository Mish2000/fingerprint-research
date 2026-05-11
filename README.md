# Fingerprint Research Platform

> A full-stack biometric research system for fingerprint verification, 1:N identification, benchmark comparison, and live scanner-backed demos.

<p align="center">
  <img src="docs/assets/readme/benchmarks.png" alt="Benchmark Explorer screenshot" width="100%" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688">
  <img alt="React" src="https://img.shields.io/badge/UI-React%20%2B%20TypeScript-61DAFB">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/DB-PostgreSQL%20%2B%20pgvector-336791">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
</p>

This repository contains an end-to-end fingerprint biometrics platform built as a research-grade system rather than a single notebook or isolated model. It combines classical computer vision, pretrained deep embeddings, method-aware 1:N retrieval, benchmark provenance, a polished React interface, and optional scanner integration for stakeholder demos.

## What this project demonstrates

- **1:1 verification** with curated demo cases, dataset browsing, manual upload, score/threshold reporting, latency breakdowns, and match overlays.
- **1:N identification** with enrollment, search, deletion, isolated browser-mode galleries, operational controls, vector shortlisting, and pairwise reranking.
- **Validated benchmark exploration** with champion cards, comparison tables, artifact links, run provenance, canonical vs. research method separation, and deterministic winner selection.
- **Method registry discipline**: API names, UI labels, aliases, benchmark names, thresholds, retrieval capability, rerank capability, vector dimensions, and presentation tier are centralized in `configs/methods.yaml`.
- **Privacy-aware storage design**: identity metadata and biometric templates can be split across separate PostgreSQL databases, with explicit inspection/reconciliation endpoints and no new raw fingerprint image persistence in the identification store.
- **Hardware integration path** for a Biometrika HiScan-Pro scanner through a Windows TWAIN helper and a saved-file bridge fallback.

## UI preview

| Live demo | Verification | Identification |
|---|---|---|
| <img src="docs/assets/readme/demo.png" alt="Live Demo" width="320" /> | <img src="docs/assets/readme/verification.png" alt="Verification" width="320" /> | <img src="docs/assets/readme/identification.png" alt="Identification" width="320" /> |

The UI is designed as a stakeholder-facing research interface: default landing on Live Demo, separate verification and identification workspaces, benchmark evidence, local preferences, theme/language support, and clear operational states.

## System architecture

```mermaid
flowchart LR
    UI[React + TypeScript UI] --> API[FastAPI backend]

    API --> Match[MatchService]
    API --> Identify[IdentificationService]
    API --> Bench[Benchmark catalog]
    API --> Scanner[Scanner capture bridge]

    Match --> Classic[Classic CV matchers\nORB / GFTT+ORB / Harris / SIFT]
    Match --> Deep[Pretrained embedding baselines\nResNet / ViT]
    Match --> Dedicated[Dedicated patch descriptor\nexperimental]

    Identify --> Retrieval[pgvector shortlist retrieval]
    Identify --> Rerank[Pairwise reranking]
    Identify --> Store[Secure split store]

    Store --> BioDB[(Biometric DB\ntemplates + vectors)]
    Store --> IdentityDB[(Identity DB\nmetadata + mappings)]

    Bench --> Artifacts[CSV / JSON / ROC / run manifests]
    Scanner --> TWAIN[Windows TWAIN helper]
    Scanner --> Bridge[Saved-file import bridge]
```

## Core workflows

### Verification, 1:1

The verification workspace compares two fingerprint images and returns a decision-oriented response: score, threshold, latency, method metadata, and optional visual overlay matches. It supports curated demo cases, catalog-backed dataset browsing, and manual upload without relying on hidden local filesystem assumptions in the browser.

### Identification, 1:N

The identification workflow separates fast shortlist retrieval from final reranking. Retrieval-capable methods can generate indexed vectors for candidate shortlist search, while rerank-capable methods can perform pairwise scoring against shortlisted candidates. This allows the system to handle method capability honestly instead of pretending every algorithm is suitable for direct vector retrieval.

Available modes:

- **Demo Mode**: curated, server-backed gallery and probe stories.
- **Browser Mode**: isolated catalog-backed gallery selection for repeatable 1:N experiments.
- **Operational Mode**: lower-level enroll, search, and delete controls.

### Benchmark Explorer

The benchmark workspace presents validated evidence rather than only raw metric tables. It separates canonical showcase methods from research/experimental rows, exposes artifact provenance, and supports best-accuracy, lowest-error, and latency-oriented views.

Current curated benchmark artifacts include `results_summary.csv`, `results_summary.md`, `run_manifest.json`, and `validation.ok` under `artifacts/reports/benchmark/current/`.

### Live Demo

The Live Demo is a guided stakeholder flow: enrollment capture, separate probe capture, and an Identify 1:N result hero. It intentionally separates enrollment and probe sources to avoid same-image matching in demos.

### Scanner capture

The scanner layer supports two capture providers:

- **Direct TWAIN capture** through `tools/biometrika_capture/bin/x86/biometrika_twain_capture.exe`.
- **Saved-file bridge** for importing the latest scan produced by external diagnostic tooling.

The backend exposes scanner status, capture, latest-import, and normalized asset endpoints. The UI includes direct capture, scanner-UI capture, import-latest, manual upload, status reporting, and a countdown before headless capture.

## Recognition methods

| Method | Runtime role | Benchmark role | Status |
|---|---|---|---|
| Classic ORB | 1:1 verification, 1:N retrieval, rerank | API/runtime classic baseline | Canonical |
| Classic ROI GFTT+ORB | 1:1 verification, 1:N retrieval, rerank | `classic_v2` benchmark family | Canonical |
| Harris + ORB | 1:1 verification, 1:N retrieval, rerank | `harris` | Canonical |
| SIFT | 1:1 verification, 1:N retrieval, rerank | `sift` | Canonical |
| Pretrained CNN embedding | 1:1 verification, vector retrieval, rerank | `dl_quick` | Canonical baseline |
| ViT embedding | 1:1 verification, vector retrieval, rerank | `vit` | Canonical baseline |
| Dedicated Patch AI | Pairwise research rerank | `dedicated` | Experimental / research |
| Fusion Balanced v1 | Benchmark-only score fusion | `fusion_balanced_v1` | Research artifact, not a production API method |

The registry explicitly prevents unsupported method usage. For example, experimental methods that are valid for reranking are not automatically advertised as direct retrieval methods unless a scientifically validated global retrieval vector exists.

## Benchmark snapshot

Current benchmark snapshot: **NIST SD300b**, `test` split, **2,844 pairs**. These values are included to document current research behavior and trade-offs, not to claim production biometric certification.

| Method | Track | AUC ↑ | EER ↓ | TAR @ FAR 1e-2 ↑ | Wall time / pair |
|---|---|---:|---:|---:|---:|
| Fusion Balanced v1 | Research fusion | 0.7294 | 0.3558 | 0.3291 | 0.02 ms* |
| SIFT | Canonical | 0.6668 | 0.3343 | 0.3291 | 33.88 ms |
| Pretrained CNN baseline | Canonical | 0.6003 | 0.4320 | 0.0197 | 14.33 ms |
| ViT baseline | Canonical | 0.5971 | 0.4325 | 0.0309 | 18.32 ms |
| Dedicated Patch AI | Research / experimental | 0.5581 | 0.4613 | 0.0098 | 159.42 ms |
| Classic ROI GFTT+ORB | Canonical | 0.5111 | 0.4890 | 0.0127 | 16.21 ms |
| Harris + ORB | Canonical | 0.5027 | 0.4972 | 0.0042 | 276.49 ms |

\* Fusion is a score-level benchmark artifact over existing method scores. Its wall-time value is not comparable to a full image-pair matcher runtime.

Key interpretation: SIFT is currently the strongest canonical single-method performer on the shown NIST SD300b test split, while fusion experiments are tracked separately as research artifacts.

## Tech stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.11, FastAPI, Pydantic, Uvicorn |
| Computer vision | OpenCV, NumPy, SIFT/ORB/Harris/GFTT, geometric verification |
| Deep learning | PyTorch, torchvision pretrained backbones |
| Storage | PostgreSQL, pgvector, optional split biometric/identity databases |
| Frontend | React, TypeScript, Vite, Tailwind CSS, lucide-react |
| Testing | pytest, FastAPI TestClient, Vitest, TypeScript contract tests |
| Scanner path | Windows TWAIN helper, saved-file bridge fallback |

## Repository layout

```text
apps/
  api/                         FastAPI application, schemas, services, scanner bridge, benchmark catalog
  ui/                          React + TypeScript stakeholder interface
configs/
  datasets.yaml                Dataset registry and manifest expectations
  methods.yaml                 Central method registry and capability contract
  thresholds.yaml              Decision thresholds and benchmark defaults
docs/
  benchmark_methods.md         Benchmark/method documentation
  scanner_capture.md           Scanner capture setup and troubleshooting
  assets/readme/               README screenshots
pipelines/
  benchmark/                   Evaluation runners, validation, cache contracts
scripts/                       Local orchestration and diagnostics
src/fpbench/
  identification/              Secure split store and identification primitives
  matchers/                    Classic, deep, and experimental matchers
  preprocess/                  Image preprocessing utilities
tests/                         Backend, storage, identification, registry, benchmark tests
```

Large local datasets, generated benchmark outputs, caches, checkpoints, scanner captures, and archive folders are intentionally excluded from normal Git tracking.

## Running locally

### 1. Create the Python environment

```bash
conda env create -f environment.yml
conda activate fingerprint_research
```

### 2. Start local PostgreSQL + pgvector databases

```bash
docker compose -f apps/api/docker-compose.yml up -d biometric_db identity_db
```

For local development, configure database URLs through environment variables:

```bash
export DATABASE_URL="postgresql://admin:change_me_biometric_dev_password@127.0.0.1:5432/biometric_db"
export IDENTITY_DATABASE_URL="postgresql://admin:change_me_identity_dev_password@127.0.0.1:5433/identity_db"
```

PowerShell equivalent:

```powershell
$env:DATABASE_URL = "postgresql://admin:change_me_biometric_dev_password@127.0.0.1:5432/biometric_db"
$env:IDENTITY_DATABASE_URL = "postgresql://admin:change_me_identity_dev_password@127.0.0.1:5433/identity_db"
```

If `IDENTITY_DATABASE_URL` is omitted, the store can fall back to a single-database deployment.

### 3. Run the API

```bash
uvicorn apps.api.main:app --reload
```

Useful endpoints:

```text
GET    /api/health
GET    /api/methods
POST   /api/match
GET    /api/benchmark/summary
GET    /api/benchmark/comparison
GET    /api/benchmark/best
GET    /api/catalog/datasets
GET    /api/catalog/dataset-browser
GET    /api/identify/stats
POST   /api/identify/enroll
POST   /api/identify/search
GET    /api/scanner/status
POST   /api/scanner/capture
```

### 4. Run the UI

```bash
cd apps/ui
npm install
npm run dev
```

For a production build:

```bash
npm run build
```

## Testing

Backend tests:

```bash
python -m pytest tests/test_method_registry_api.py tests/test_identification_pipeline.py tests/test_identification_admin_api.py tests/test_benchmark_api_stage3.py -q
```

Storage and migration tests:

```bash
python -m pytest tests/test_secure_split_store.py tests/test_secure_split_store_migration_postgres.py -q
```

Frontend contract and UI tests:

```bash
cd apps/ui
npm test
```

The test suite covers method registry behavior, API aliases, benchmark catalog repair/validation, path traversal guards for benchmark artifacts, identification retrieval/rerank capability enforcement, secure split-store migrations, lazy service initialization, and frontend catalog/file-loading behavior.

## Data and artifact expectations

The repository is designed to work with local datasets and generated artifacts without committing large or sensitive data to Git.

Typical local paths:

```text
data/raw/                         Original datasets, local only
data/manifests/<dataset>/          Manifest and pair protocol files
data/processed/<dataset>/          Normalized processed assets, local only
artifacts/reports/benchmark/       Benchmark summaries, scores, ROC plots, manifests
artifacts/cache/embeddings/        Embedding cache, local only
artifacts/checkpoints/             Model checkpoints, local only
```

The dataset registry currently includes active and optional research datasets such as NIST SD300B/SD300C, PolyU cross-modality, PolyU 3D, UNSW 2D/3D, and L3-SF V2. Dataset files must be obtained and prepared separately according to their original terms and the project ingest pipeline.

## Privacy and security posture

This project is a research prototype, but several implementation choices are intentionally production-aware:

- Biometric data and identity metadata can be separated into different PostgreSQL databases.
- Admin inspection endpoints redact sensitive database URLs.
- Identification diagnostics expose schema/readiness state without requiring UI-side secrets.
- New identification enrollments are designed around metadata/templates/vectors rather than durable raw-image persistence.
- Scanner captures are normalized as runtime artifacts and should be treated with explicit retention policy.
- Benchmark and catalog asset routes include path traversal protection.

## Current limitations

- This is a research and demonstration system, not a certified biometric product.
- Current benchmark results are transparent research metrics and should not be interpreted as production FAR/FRR guarantees without threshold calibration on the target deployment population.
- Large datasets, model checkpoints, scanner captures, and generated artifacts are intentionally local and may not be available after a fresh clone.
- Direct TWAIN capture is Windows-specific and depends on a registered 32-bit TWAIN stack and the expected Biometrika source name.
- Dedicated Patch AI is intentionally kept in the experimental/research track until its retrieval and benchmark behavior justify promotion.
- Fusion Balanced v1 is a benchmark score-fusion artifact, not a normal runtime method exposed by the main API.

## Author & Research Supervision

**Michael Sirkovich**
Computer Science Graduate
Project development, system architecture, implementation, benchmarking, and UI/API integration.

**Academic Advisor:** Prof. Menachem Domb
Research guidance and academic supervision.
