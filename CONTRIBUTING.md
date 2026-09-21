# Contributing

Keep changes focused on the workbench's existing React, FastAPI, PostgreSQL and
Java components. A software integration change does not authorize a new research
experiment, new cohort, training run, threshold search or score optimization.

Use a `codex/` feature branch and a pull request. Run the relevant local checks,
review the actual staged filenames and diff, and merge only after CI succeeds.
Verify CI on the resulting `main` commit before removing the merged branch.
Do not bypass branch protection or rewrite published history.

## Local tests

Use the prepared CPU environment. A regular shell can invoke its Python by full
path, or use a Miniconda prompt with `fingerprint_research_cpu` activated.

```powershell
python -m pip check
python -m pytest tests -q -ra
cd apps/ui
npm test
npm run lint
npm run build
```

For Windows long-path tests, pass an explicit extended temporary path, for
example `--basetemp=\\?\C:\your-checkout\.local\pytest`. Pytest may remove this
temporary directory; use a dedicated test directory only.

The [runbook](docs/LOCAL_DUAL_DB_RUNBOOK.md) describes real PostgreSQL and Java
integration. When enabled, unavailable required services fail the tests. The
default suite does not download weights. For offline unit checks on a workstation
that already has weights, set `FPBENCH_WEIGHTS_DIR` to an empty test directory.

Four historical-evidence test modules require original private report bundles.
They are explicitly marked `historical_artifacts`. Set
`FPBENCH_HISTORICAL_TESTS=true` to require those files and execute their unchanged
evidence assertions. Missing artifacts are not a successful reproduction.

## Scientific and privacy boundaries

- Preserve existing report, catalog, image, score, protocol and split files.
  Their historical run identities do not become the current source revision.
- Never commit `.env`, local paths, credentials, captures, templates, embeddings,
  checkpoints, raw datasets or private review reports. `.local/` is ignored.
- New public fixtures and screenshots must be synthetic, with no human source.
  Existing tracked research assets are not a precedent for publishing more data.
- Keep score zero distinct from failure. Report missing resources explicitly;
  do not substitute a different model or provider.
- Preserve external authorship, licenses and dataset terms. This repository does
  not create a blanket license for third-party code, data or weights.

The retired dedicated patch method retains legacy metadata/source for historical
artifact interpretation. It is excluded from active API methods and UI choices.
