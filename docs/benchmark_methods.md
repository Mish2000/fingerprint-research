# Benchmark Method Tiers

The benchmark stack separates canonical showcase methods from research methods.

Canonical methods are eligible for default benchmark matrix runs, benchmark catalog champion cards, and the main comparison story when their artifacts pass validation. The current canonical benchmark defaults are `classic_v2`, `minutiae`, `harris`, `sift`, `dl_quick`, and `vit`.

`minutiae` is the classical crossing-number minutiae baseline. It uses polarity-validated ridge enhancement, deterministic Zhang-Suen skeletonization, crossing-number endings/bifurcations, stronger dense-junction pruning, and alignment-based pairwise matching. Its 512D aggregate vector is for shortlist retrieval only; pairwise minutiae alignment remains the authoritative score.

The current `decision.api.minutiae` threshold is not a calibrated FAR/FRR/EER operating point. The v2 semantics epoch (`minutiae_crossing_number_aligned_v2`) adds extraction-quality diagnostics, ridge-polarity validation, saturation and dominance flags, and quality-penalized scoring so dense random templates are not rewarded simply for producing many tentative correspondences. Minutiae extraction and skeleton matching are more expensive than the ORB/Harris/SIFT retrieval-vector paths, so automated tests should use tiny `--limit` smoke runs and full benchmark runs should be launched manually only when calibration artifacts are intended.

Dedicated Patch AI remains available as an experimental research method. It is not included in canonical benchmark defaults and is not eligible for champion or showcase selection while `configs/methods.yaml` marks `showcase_eligible: false`.

Run Dedicated Patch AI only through an explicit research request, for example:

```powershell
python pipelines/benchmark/run_benchmark_matrix.py --profile research
python pipelines/benchmark/run_benchmark_matrix.py --methods dedicated --splits val
python pipelines/benchmark/validate_benchmark_bundle.py --profile dedicated --outdir artifacts/reports/benchmark/dedicated_audit --expected_splits val
```

Promotion to canonical requires reproducible benchmark evidence that satisfies the promotion criteria in `configs/methods.yaml`: matching or beating canonical baselines on agreed bundles, passing full NIST SD300b/SD300c and PolyU gates, stable AUC/EER/TAR@FAR with acceptable latency, and validated run metadata/artifacts.
