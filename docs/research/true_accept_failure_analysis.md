# True-accept failure analysis across methods

This diagnostic answers two supervisor questions:

1. Can we identify the positive pairs that did not become true accepts and recompute TAR after removing them?
2. Are those missed positive pairs identical across methods, or can one method accept a pair that another method rejects?

## Interpretation

For an individual positive pair, the method outcome is `TA` or `FR`.
`TAR` is the aggregate rate:

```text
TAR = TA / positives
```

Therefore, the cases with no TAR answer at the pair level are interpreted as positive pairs with:

```text
label = 1
score < threshold
outcome = FR
```

## Important validity note

Removing false-rejected positives from TEST is diagnostic only. It is not a valid benchmark result, because the removed samples are selected after observing the method outcome.

The valid benchmark remains the original TAR/FAR measured on the full TEST split. The filtered recomputation is only a sanity check and failure-analysis aid.

## Outputs

The script writes:

```text
all_method_outcomes.csv
method_outcome_summary.csv
positive_pair_outcome_matrix.csv
common_false_rejects_all_methods.csv
method_specific_false_rejects.csv
pairwise_complementarity_summary.csv
current_diagnostics_manifest.json
true_accept_failure_summary.md
```

The main file for the supervisor is:

```text
true_accept_failure_summary.md
```

## Recommended command

```powershell
python scripts\diagnostics\build_current_fusion_v2_diagnostics.py `
  --repo-root C:\fingerprint-research
```

## Meaning of selected outputs

- `common_false_rejects_all_methods.csv`: positives missed by all selected methods.
- `method_specific_false_rejects.csv`: positives missed by one method but accepted by at least one other method.
- `pairwise_complementarity_summary.csv`: for every method pair, counts how many positives one method rescues relative to another.

The older cross-benchmark version of this analysis is quarantined under `artifacts/reports/diagnostics/legacy_stale_20260629/` because it does not correspond to the current canonical Fusion v2 statistical run.

## Custom methods

If a method is not part of the built-in aliases, pass a custom spec:

```powershell
--method-spec my_alias=relative_or_absolute_benchmark_dir:method_id `
--methods fusion_v2,my_alias
```

For relative benchmark directories, the script resolves them under:

```text
artifacts/reports/benchmark/
```
