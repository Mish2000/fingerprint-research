# Vector Reproducibility Final Summary

Source folder: `artifacts/reports/identification/vector_reproducibility_demo`

## Scope

- Purpose: deterministic vectorization proof for retrieval-vector storage and method-generic vector architecture.
- Dataset/image metadata: `nist_sd300b`, subject `1000`, plain impression, FRGP `2`, PPI `1000`, manifest row index `1`, valid sample index `1`, filename `00001000_plain_1000_02.png`.
- Methods: `classic_gftt_orb`, `minutiae`, `harris`, `sift`, `dl`.
- Procedure: vectorize the same fingerprint image twice with the same retrieval method and configuration, then compare the resulting float32 vectors numerically, by SHA-256, and by binary representation.
- Epsilon: `1e-06`.
- Preview tables: retained in the source folder only; they are intentionally not copied into this final summary.

## Determinism Evidence

| method | vector dim | exact equal | allclose equal | max abs diff | nonzero diffs | binary equal dimensions | binary equal bits |
|---|---:|---|---|---:|---:|---:|---:|
| classic_gftt_orb | 512 | true | true | 0 | 0 | 512/512 | 16,384/16,384 |
| minutiae | 512 | true | true | 0 | 0 | 512/512 | 16,384/16,384 |
| harris | 512 | true | true | 0 | 0 | 512/512 | 16,384/16,384 |
| sift | 512 | true | true | 0 | 0 | 512/512 | 16,384/16,384 |
| dl | 512 | true | true | 0 | 0 | 512/512 | 16,384/16,384 |

Each full comparison CSV contains `512` dimensions with maximum absolute difference `0`. Each binary comparison CSV contains `512` dimensions with `0` non-equal binary rows.

## SHA-256 Evidence

| method | vector 1 SHA-256 | vector 2 SHA-256 |
|---|---|---|
| classic_gftt_orb | `7a518e647b05feaa296ebf54fe5e65564a0c39600f9b024a1ef8054b68e2e6a1` | `7a518e647b05feaa296ebf54fe5e65564a0c39600f9b024a1ef8054b68e2e6a1` |
| minutiae | `e26e635931d844ec9dfd6d95138db472e696007e4be755358b1f6d6b0328466e` | `e26e635931d844ec9dfd6d95138db472e696007e4be755358b1f6d6b0328466e` |
| harris | `c95164220e97282c203434f988f1f7c64f6a23f67eb9fc30ff93dfdf714b1b14` | `c95164220e97282c203434f988f1f7c64f6a23f67eb9fc30ff93dfdf714b1b14` |
| sift | `7eaf2e39a66f980282bf3539a178c287e81cd3844dd9b038f9ac6660aa04568d` | `7eaf2e39a66f980282bf3539a178c287e81cd3844dd9b038f9ac6660aa04568d` |
| dl | `f53c2081bf5b384289c25b7388e7a80543c0e02692dc51faad8b6639dfbdf1d5` | `f53c2081bf5b384289c25b7388e7a80543c0e02692dc51faad8b6639dfbdf1d5` |

## Why It Matters

This proves that the retrieval vector layer can store deterministic method outputs without relying on method-specific table shapes. That supports method-generic pgvector storage, reproducible retrieval, and clean separation between image identity, retrieval vectors, and optional downstream reranking.

Final recommendation: keep this as vector determinism evidence.
