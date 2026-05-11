# Benchmark Method Tiers

The benchmark stack separates canonical showcase methods from research methods.

Canonical methods are eligible for default benchmark matrix runs, benchmark catalog champion cards, and the main comparison story when their artifacts pass validation. The current canonical benchmark defaults are `classic_v2`, `harris`, `sift`, `dl_quick`, and `vit`.

Dedicated Patch AI remains available as an experimental research method. It is not included in canonical benchmark defaults and is not eligible for champion or showcase selection while `configs/methods.yaml` marks `showcase_eligible: false`.

Run Dedicated Patch AI only through an explicit research request, for example:

```powershell
python pipelines/benchmark/run_benchmark_matrix.py --profile research
python pipelines/benchmark/run_benchmark_matrix.py --methods dedicated --splits val
python pipelines/benchmark/validate_benchmark_bundle.py --profile dedicated --outdir artifacts/reports/benchmark/dedicated_audit --expected_splits val
```

Promotion to canonical requires reproducible benchmark evidence that satisfies the promotion criteria in `configs/methods.yaml`: matching or beating canonical baselines on agreed bundles, passing full NIST SD300b/SD300c and PolyU gates, stable AUC/EER/TAR@FAR with acceptable latency, and validated run metadata/artifacts.
