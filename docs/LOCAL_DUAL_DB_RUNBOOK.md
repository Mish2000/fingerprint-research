# Local Dual-Database Runbook

The supported Windows hybrid runtime uses Docker for two PostgreSQL/pgvector
services and the host for Python, Java and Node. CPU is the default. Use
fictitious identities and synthetic fixtures for the software demo.

## Preparation

Start the existing Docker Desktop with Linux containers and ensure Node.js
24.18.0 is available. From the checkout root:

```powershell
.\scripts\dev\bootstrap.ps1
.\scripts\dev\workbench.ps1 init-config
.\scripts\dev\workbench.ps1 build-java
.\scripts\dev\workbench.ps1 prepare-models
.\scripts\dev\workbench.ps1 check
```

Bootstrap creates a new environment from `requirements/windows-native.explicit.txt`
and the separate pip lock. Existing environments are preserved. Supply
`-CondaExe` or a fresh `-EnvironmentName` when needed. The previous research
environment is not upgraded in place.

The workbench wrapper locates Python without shell activation. It accepts
`-EnvironmentName` or `-Python` and passes that interpreter explicitly to child
services. Java resolves to the actual JVM executable, including Conda's
`Library/lib/jvm/bin` location. `environment.yml` is a CPU recipe; the explicit
lock/bootstrap path pins the tested Windows builds.

## One configuration path

`init-config` creates ignored `.env` with random private passwords, two complete
DB URLs, a named Compose project and loopback ports. It does not print passwords
or overwrite files. `.env.example` is the public reference.

The loader accepts literal `KEY=value` entries and optional matching quotes.
There is no shell evaluation or interpolation. **Process environment values win
over file values.** The PowerShell wrapper explicitly selects Python. The Python
launcher loads `.env` once and passes the same values to Compose, Java, API and UI.

A direct API launch can opt in through `FPBENCH_ENV_FILE`. Ordinary imports and
unit tests do not search for a private environment file.

The demo requires distinct, local `DATABASE_URL` and `IDENTITY_DATABASE_URL`,
`SOURCEAFIS_ENABLED=true`, `SOURCEAFIS_SERVICE_URL` and
`FINGERPRINT_ENGINE_PROVIDER=sourceafis_open`. Omitting the identity URL is not
accepted as a split-database demonstration.

| Resource | Demo | Isolated tests |
|---|---|---|
| Compose project | fingerprint-research-demo | fingerprint-research-test |
| Biometric PostgreSQL | 127.0.0.1:55432 | 127.0.0.1:55434 |
| Identity PostgreSQL | 127.0.0.1:55433 | 127.0.0.1:55435 |
| API / UI / Java ports | 8000 / 5173 / 8765 | 8001 / 5174 / 8766 |

Each project owns two separate volumes. Compose uses a digest-pinned pg16 image,
loopback bindings, `pg_isready` healthchecks and `up --wait`. Docker's credential
helper is found through child PATH only. No other WSL distribution is changed.

## Start and stop

```powershell
.\scripts\dev\workbench.ps1 start
# Open http://127.0.0.1:5173/?tab=verify
.\scripts\dev\workbench.ps1 stop
```

Start checks prerequisites and ports, then waits for databases, Java health,
API readiness and UI. Process IDs, creation times, executable identities and
logs live under ignored `.local/`. Existing process state requires `stop`
before another start. A conflicting process is not terminated automatically.

Stop terminates only recorded process trees and stops that project's containers.
**It preserves volumes.** Never use a global Docker prune to repair the demo.

`/live` is process liveness. `/ready` returns 503 until methods, both DB connections
and SourceAFIS are ready. `/methods` reports loaded availability. `/health` retains
legacy diagnostics; lazy startup is a test facility, not full readiness.

## Models and CPU/CUDA

Preparation fetches exactly `ResNet18_Weights.IMAGENET1K_V1` and
`ViT_B_16_Weights.IMAGENET1K_V1` from the official PyTorch model host. Local files
and SHA-256 manifests are stored in ignored `artifacts/checkpoints/torchvision`,
or `FPBENCH_WEIGHTS_DIR`. Corrupt/mismatched files fail validation. Runtime
loading is offline only; page navigation never starts a download.

Input size, preprocessing and existing thresholds are unchanged. These are
ImageNet backbones, not fingerprint models trained here. The original dedicated
checkpoint is unavailable and that active method is retired.

Create a separate optional GPU environment:

```powershell
.\scripts\dev\bootstrap.ps1 -Profile cuda128 -EnvironmentName fingerprint_research_gpu
$env:FPBENCH_DEVICE = 'cuda'
.\scripts\dev\workbench.ps1 start -EnvironmentName fingerprint_research_gpu
.\scripts\dev\workbench.ps1 stop -EnvironmentName fingerprint_research_gpu
Remove-Item Env:FPBENCH_DEVICE
```

Stop the CPU demo before changing profiles. CUDA selection never silently falls
back to CPU. The pinned pair is PyTorch 2.7.1 CUDA 12.8 / torchvision 0.22.1; see
[Blackwell support](https://pytorch.org/blog/pytorch-2-7/) and the official
[versioned installation commands](https://pytorch.org/get-started/previous-versions/).
Local tests exercised both models on `cuda:0` with finite 512D/768D outputs.
CPU/GPU numerical parity is not claimed. Unused `facenet-pytorch` is excluded
from the new profiles; the old environment remains unchanged.

## Enrollment, retention and reset

Use the Identification workspace with a synthetic image, fictitious name and
demo identifier. Select retrieval vectors explicitly. ResNet uses 512D, ViT 768D;
classical retrieval adapters retain their existing 512D contracts.
`sift_plain_roll_v2` is pairwise rerank-only.

PostgreSQL stores metadata, vectors and identity links. The demo retains inputs
under `.local/<profile>/enrollments`, keyed by content hash, so upload reranking
can survive restart. This is explicit local retention, not encrypted template
protection or a production security guarantee. Identity deletion removes DB
links and the local input after its final reference in that store is removed.

The existing Reset demo / Reset browser store actions target their seeded
namespaces. Uploaded people are deleted individually. No reset targets an
unrelated database or deletes Docker volumes.

## Data and historical reports

Synthetic fixtures need no raw dataset. Curated catalog/report assets can be
read without `data/raw` or original manifests.

Set `FPBENCH_DATASETS_ROOT` to the actual external dataset root when needed.
The read-only resolver maps `data/raw/sd300b` to `NIST/sd300b`, `sd300c` to
`NIST/sd300c`, and `PolyU_Hong_Kong` to `PolyU Hong Kong`; other directory names
are preserved. It creates no junction and writes no source cache.

Original manifest families for SD300B, SD300C, PolyU Cross, PolyU 3D, UNSW 2D/3D
and L3-SF V2 remain incomplete: 87 declared source files are missing. Eight
selected-pair CSVs in retained final bundles are preserved, but do not recover
all original splits. Reproduction requires the originals. A new selection must
never be described as a restoration.

## Scanner

Build with an existing Visual Studio C++ installation and Windows SDK:

```powershell
.\tools\biometrika_capture\build_x86.ps1
```

The launcher uses the new `.local/scanner/biometrika_twain_capture.exe` when
present. `SCANNER_TWAIN_HELPER_PATH` overrides the path. The previously tracked
binary is preserved. Build discovery supports current VS/SDK locations.

`/scanner/status` reports driver discovery. Physical capture additionally needs
hardware and a participating user. The local x86/TWAIN path was verified with
the Biometrika driver and a real capture. This is a software/device check,
not a liveness or accuracy evaluation.

For file import, set `SCANNER_CAPTURE_DIR` and explicitly select
`saved_file_bridge`. `SCANNER_NORMALIZED_DIR` and `SCANNER_TWAIN_RAW_DIR` control
private outputs. Launcher defaults place these under its runtime directory.
Import is never reported as fresh physical capture; fallback requires an explicit
user choice.

## Validation

Create separate test resources:

```powershell
.\scripts\dev\workbench.ps1 init-config -EnvFile .env.test -TestProfile
.\scripts\dev\workbench.ps1 db-up -EnvFile .env.test
```

Set `IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL` and
`IDENTIFICATION_TEST_IDENTITY_DATABASE_URL` to those test databases.
`FPBENCH_INTEGRATION_REQUIRED=true` makes missing required configuration fail.
Set `SOURCEAFIS_INTEGRATION_TESTS=true` and a live service URL for the Java
roundtrip. `FPBENCH_RUNTIME_INTEGRATION_TESTS=true` enables real API/DB/Java tests.
Never point migration tests at an application database.

Follow [CONTRIBUTING](../CONTRIBUTING.md) for test/build/lint commands. Core CI
uses synthetic fixtures without private weights, raw datasets or hardware.
Historical checks missing original files are skipped and reported separately.
A clean install uses a new environment, `npm ci` and a fresh Java build; download
caches may be reused without relying on existing build outputs.

## Recovery

- Docker client without daemon: start the installed Desktop. User-required setup
  screens must be handled by the user, without reinstalling Docker unnecessarily.
- Missing weights: run explicit preparation, then stop/start. Health does not download.
- DB/Java interruption: requests remain failures and readiness drops. Stop/start
  restores owned services with existing volumes; never kill all Java processes.
- Bad images/missing DPI: fix the input. Decode errors do not become match scores.
- Port/PID conflict: stop the correct owned demo; unrelated processes are preserved.
