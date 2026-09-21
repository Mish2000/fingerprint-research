# Engineering case study: a local fingerprint workbench

The application joins interactive computer vision, model inference, two database
roles and a Java matching engine. The engineering problem was making those
existing pieces work together reproducibly on a new Windows workstation while
preserving historical research results.

## What is implemented here

- A React/TypeScript interface for upload verification, sample browsing,
  enrollment, identification and report inspection.
- A FastAPI layer that resolves method aliases, enforces capability contracts,
  handles image inputs and keeps retrieval distinct from pairwise reranking.
- PostgreSQL/pgvector storage with separate biometric and identity databases,
  schema inspection, migration checks and explicit deletion.
- Python adapters for classical CV and official ImageNet ResNet/ViT backbones.
- An HTTP integration with the external SourceAFIS Java engine, including
  extraction, 1:1 verification and provider-native gallery identification.
- Windows TWAIN capture software and an explicit saved-file import bridge.

SourceAFIS supplies the external matcher; torchvision supplies the pretrained
backbones. Their algorithms and ImageNet training are not claimed as work
performed in this repository. Local CV composition, service contracts, orchestration,
storage, UI integration and software validation are the implementation focus.

## Changes that made the runtime usable

Model preparation is an explicit command. Runtime loading checks the local
weight file and manifest, uses safe state-dictionary loading and never starts a
download. CPU is the default, and CUDA requires an explicit selection. The GPU
profile runs in a separate Conda environment from both CPU and the old installation.

One literal configuration file feeds Compose and all child services. Two explicit
database URLs are required by the demo. Containers bind only to loopback and
readiness checks query both database roles, including an empty gallery.

The existing store deliberately retains image metadata rather than image bytes.
Uploaded enrollment images therefore needed an explicit local source for reranking.
The demo now keeps content-addressed input files outside PostgreSQL, verifies their
SHA-256 before use and removes them when their final database reference is deleted.
Names and identity fields remain in the identity database.

The Python/Java boundary carries explicit DPI and raw scores. Service failure
cannot become a non-match or a score of zero. The UI exposes a separate SourceAFIS
comparison instead of treating engine metadata as a match result.

Process supervision uses the actual Java executable, records PID, creation time
and executable identity, and stops only owned process trees. This addresses the
Conda Java launcher's otherwise surviving JVM child. Stopping the demo preserves
its database volumes.

## Validation and limits

Validation separates unit tests with substitutes, real PostgreSQL/Java integration,
actual model inference, browser interactions and historical artifact checks.
Synthetic software fixtures exercise the live system without creating a research
population. CPU and RTX 5080 inference produce finite 512D/768D vectors; this is
compatibility evidence, not a claim of CPU/GPU numerical identity or accuracy.

The original dedicated checkpoint was unavailable and that application method
was retired. Original manifest families remain incomplete; retained selected-pair
files do not reconstruct every original split. Historical reports remain available
and retain their own provenance. No research benchmark was rerun for this work.

See the [capability status](../README.md#capabilities),
[runtime and test instructions](LOCAL_DUAL_DB_RUNBOOK.md), and
[contribution boundaries](../CONTRIBUTING.md).
