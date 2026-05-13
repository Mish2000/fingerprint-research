# Central Demo Catalog

`data/samples/catalog.json` is the single source of truth for curated demo-ready fingerprint cases.

## What this catalog contains

- `verify_cases`: curated 1:1 verification stories with traceability back to `pairs_<split>.csv` and manifest rows.
- `identify_gallery`: curated identities plus scenario seeds for positive, difficult, and no-match 1:N demos.
- `dataset_browser_seed`: a small, deterministic, diverse subset per dataset for future browsing UX.
- `source_datasets`: authoritative provenance describing which manifests, stats, split metadata and benchmark runs were used.
- `data/samples/assets/`: deterministic local PNG assets plus thumbnails for every curated asset referenced by the catalog.
- `verify_cases` intentionally stay narrow and curated; broader datasets can still appear under identity and browser coverage without being verify-demo-exposed.

## Source of truth and relation to existing project data

- The catalog is derived from `data/manifests/<dataset>/manifest.csv`, `pairs_*.csv`, `stats.json`, `split.json`, and `protocol_note.md` when present.
- Where benchmark evidence exists, recommended methods and difficulty ordering are anchored to `artifacts/reports/benchmark/...`.
- `catalog.json` is **not** a raw dump of manifests. It is a curated layer that references official project artifacts and adds deterministic demo semantics.
- `processed/` and raw corpora remain upstream storage layers; the catalog preserves them via `source_path` / `traceability` while the consumer-facing `path` points to the local curated asset layer.
- The local curated asset layer stores runnable binary PNG assets and thumbnails while preserving manifest-backed provenance.
- Datasets outside the 7 curated verify stories remain catalog-included for identity and browser coverage when canonical assets exist.

## Forward-compatibility contract

Consumers may rely on:

- stable top-level regions: `source_datasets`, `verify_cases`, `identify_gallery`, `dataset_browser_seed`, `metadata`
- stable IDs: `case_id`, `identity_id`, `asset_id`, `scenario_id`
- machine-verifiable structure from `catalog.schema.json`
- explicit availability semantics via `availability_status` + `availability_detail`
- `path` being a local, shipped artifact and `source_path` being the upstream provenance pointer

Consumers must **not** assume:

- that every dataset has the same finger semantics (`frgp=0` exists in some public sources)
- that every local curated asset preserves the original pixel matrix byte-for-byte; some assets are deterministically re-rendered to PNG for UI compatibility
- that `is_demo_safe=true` for real biometric datasets without an explicit usage review
- that path layout should be re-derived manually from directory structure instead of reading the catalog

## Validation summary

- Validation status: `pass`
- Errors: `0`
- Warnings: `0`
- Materialized curated assets: `55`

## Regeneration

Run:

```bash
python scripts/build_demo_catalog.py
```

This rewrites `catalog.json`, `catalog.schema.json`, `catalog.validation_report.json`, and refreshes the deterministic local asset layer under `data/samples/assets/`.
