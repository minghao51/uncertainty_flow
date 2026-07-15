# Hamilton and Medallion Benchmarking Refactor Plan

**Date:** 2026-07-13
**Status:** Review follow-up required before the migration can be declared complete; then remove the legacy benchmarking surface
**Owner:** Next implementation thread
**Scope:** Full migration of the benchmarking lifecycle described in the Hamilton Medallion Refactor Plan

## Outcome

Introduce a verifiable Hamilton-backed benchmarking lifecycle with Bronze, Silver, Gold, and Platinum artifacts while preserving the current public model contract and the existing `BenchmarkFlow`, `BenchmarkRunner`, and `uncertainty-flow benchmark` entry points during migration.

The implementation is intentionally staged. The first milestone is a complete local vertical slice; model matrices, site publishing, object storage, and scheduling follow only after that slice is verified.

## Progress log

- 2026-07-13: Phase 0 completed with compatibility fixtures, CLI/result snapshots, and ADRs 002–005.
- 2026-07-13: Phase 1 completed with immutable contracts, canonical identity helpers, local atomic artifact storage, checksum verification, and tests.
- 2026-07-13: Initial Hamilton DAG construction completed with resolved configuration, explicit source-frame input, deterministic holdout validation plans, dry-run side-effect coverage, and invalid-split coverage.
- 2026-07-13: `sf-hamilton>=1.90.0` added as the optional `benchmarking` extra; core model dependencies remain unchanged.
- 2026-07-13: The first `conformal-regressor` branch now fits, predicts, evaluates required metrics, verifies split/metric/artifact invariants, and publishes checksummed Platinum artifacts.
- 2026-07-13: Added `uncertainty-flow pipeline plan/run` CLI commands with YAML configuration and network-free local parquet execution.
- 2026-07-13: Added dataset/model/metric registries, temporal and rolling-origin split-plan modes, verified-run reuse, and explicit legacy result translation.
- 2026-07-13: Optional diagnostic requests now produce recorded `degraded` status when an adapter is unavailable; disabled diagnostics remain explicit.
- 2026-07-13: Added versioned `jsonl.gz` Platinum evidence export, checksummed `index.json`, `pipeline export-site`, and a separate Astro/Starlight evidence-site skeleton.
- 2026-07-13: Evidence-site `npm run build` passes; the existing MkDocs site remains untouched.
- 2026-07-13: Added registry-backed multi-model execution with model-specific prediction artifacts, branch isolation, degraded status, and per-model evidence records.
- 2026-07-13: Full repository verification passed: 149 tests, Ruff, mypy, and Astro/Starlight build.
- 2026-07-13: Phase 6 local hardening added run locks with stale recovery, compressed node events, conservative unverified-run GC, HMAC manifest verification, lock-aware CLI execution, and operator recovery/verification docs.
- 2026-07-13: Added provider-neutral deployment adapters for object storage, scheduling, alerts, retention planning, and local locking.
- 2026-07-13: Final verification passed: 156 tests, 5 optional skips, Ruff, mypy, and Astro/Starlight build.
- 2026-07-13: Review remediation added checksum-carrying manifest artifact references, corruption rejection during reuse/evidence export, local Bronze/Silver/Gold materialization, configurable metric/model execution, matrix reuse, and generated evidence run pages.
- 2026-07-13: `rolling_origin` is now rejected explicitly until fold-level execution and persisted multi-fold memberships are implemented; it is no longer represented as a tail holdout.

## Current baseline

The repository already provides:

- `uncertainty_flow/benchmarking/flow.py` — monolithic `BenchmarkFlow` lifecycle with loading, splitting, tuning, fit, prediction, and metrics.
- `uncertainty_flow/benchmarking/runner.py` — public runner adapter and legacy model registry.
- `uncertainty_flow/benchmarking/configs.py` — mutable dataclass configuration with legacy names.
- `uncertainty_flow/benchmarking/datasets.py` — local/HuggingFace dataset registry and loader.
- `uncertainty_flow/benchmarking/providers.py` — provider seam for built-in and legacy model adapters.
- `uncertainty_flow/benchmarking/results.py` and `sinks.py` — current result objects and JSON/CSV output.
- `uncertainty_flow/cli.py` — existing dataset, benchmark, tuning, and download commands.
- `docs/` and `site/` — MkDocs source and generated output already used by CI/GitHub Pages.
- `tests/benchmarking/` plus CLI and end-to-end tests — the initial parity/golden-fixture surface.

Important differences from the target architecture:

- No Hamilton dependency or driver/dataflow package exists yet.
- No immutable dataset/run/artifact contracts exist.
- Current run IDs are UUID fragments, not content-derived identities.
- Current flow writes final JSON/CSV outputs but has no medallion layout, staging publication, or verification manifest.
- Current configuration is benchmark-oriented and mutable; it needs a resolved immutable representation without breaking callers.
- Current documentation is MkDocs, not Astro/Starlight. The existing generated `site/` must not be overwritten during the transition.

## Decisions for this implementation

1. Keep core model packages independent of Hamilton, storage, and documentation.
2. Add the new pipeline under `uncertainty_flow/benchmarking/` and keep legacy modules as compatibility surfaces until parity is proven.
3. Use immutable Pydantic models or frozen dataclasses for metadata; keep Polars frames outside serialized contract models.
4. Use a local filesystem artifact store first, with atomic temporary-file publication and checksum verification.
5. Treat Hamilton nodes as mostly pure functions. Side effects belong in explicit storage/materialization nodes and the coordinator.
6. Use stable content-derived identities. Timestamps are metadata only and never identity inputs.
7. Required branch failure blocks publication. Optional diagnostic failure records `degraded` status with an explicit reason.
8. `plan`/dry-run must construct and validate the DAG without fetching, fitting, predicting, or writing artifacts.
9. Preserve current JSON/CSV filenames and field meanings in the first compatibility migration; translate old results into new contracts rather than silently changing them.
10. Keep MkDocs as the existing library documentation system. Introduce the Astro/Starlight evidence portal in a separate directory (recommended: `evidence-site/`) until a deliberate cutover is approved; do not repurpose the current generated `site/` directory during Phases 0–4.

## Delivery phases

### Phase 0 — Baseline and design freeze

Deliverables:

- Inventory current providers, datasets, split modes, tuning options, sinks, CLI behavior, and tests.
- Add representative local fixtures for one tabular/conformal run and one existing legacy result.
- Snapshot current `BenchmarkResult`, `ModelResult`, JSON, CSV, and CLI behavior.
- Record supported dataset/model/validation combinations and unsupported combinations.
- Add ADRs for Hamilton boundaries, medallion formats, identity/idempotency, and evidence export.
- Decide whether the first production dataset is a local parquet fixture or one pinned built-in registry dataset. The default PR gate must remain network-free.

Exit gate:

```text
uv run pytest tests/benchmarking tests/test_cli.py tests/test_full_workflow.py -q
uv run ruff check uncertainty_flow/benchmarking tests/benchmarking tests/test_cli.py tests/test_full_workflow.py
```

The baseline artifacts and compatibility expectations are reviewable before new pipeline code is introduced.

### Phase 1 — Contracts, identities, and local artifact store

Add:

```text
uncertainty_flow/benchmarking/contracts/
  datasets.py validation.py runs.py artifacts.py verification.py
uncertainty_flow/benchmarking/storage/
  base.py local.py layout.py serializers.py
uncertainty_flow/benchmarking/identity.py
```

Implement:

- `DatasetRef`, `DatasetManifest`, `DatasetSchema`, `ColumnRole`, `ValidationIssue`, `ValidatedDataset`.
- `SplitStrategy`, `SplitAssignment`, `ValidationPlan`, `RollingWindow`, `LeakageCheckResult`.
- `RunRequest`, `ResolvedRunConfig`, `RunIdentity`, `RunManifest`, `RunStatus`, and degradation reasons.
- `ArtifactRef`, artifact types/checksums, materialization results, and model artifact references.
- Verification records and run verification reports.
- Canonical serialization and deterministic SHA-256 identity helpers.
- Bronze/Silver/Gold/Platinum path resolution and schema-versioned JSON serialization.
- Local `ArtifactStore` protocol and implementation with atomic writes, read-back verification, and safe handling of incomplete staging files.

Initial identity inputs:

```text
dataset_version = hash(source checksum + ingestion contract version)
silver_version = hash(dataset version + validation contract + transformation version)
validation_plan_id = hash(silver version + split configuration)
run_id = hash(validation plan + model specification + evaluation specification + code version)
```

Exit gate:

- Contract round trips succeed.
- Identical canonical inputs produce identical IDs across processes.
- Changed configuration changes only the expected identity.
- Missing, corrupt, or partially written artifacts never verify as complete.

### Phase 2 — Single Hamilton vertical slice

Use one local dataset adapter, one representative existing model provider (recommended: `conformal-regressor`), random holdout with calibration, core probabilistic metrics, and the local artifact store.

Add:

```text
uncertainty_flow/benchmarking/driver.py
uncertainty_flow/benchmarking/coordinator.py
uncertainty_flow/benchmarking/config.py
uncertainty_flow/benchmarking/dataflows/
  ingestion.py validation.py preparation.py splitting.py
  training.py prediction.py evaluation.py calibration.py publication.py
uncertainty_flow/benchmarking/verification/
  dataset.py splits.py predictions.py metrics.py run.py
```

Initial dependency chain:

```text
run_request
  -> resolved_run_config
  -> source_dataset
  -> bronze_manifest
  -> validated_dataset
  -> silver_manifest
  -> validation_plan
  -> gold_dataset
  -> fitted_model
  -> distribution_predictions
  -> metric_results
  -> calibration_results
  -> run_verification
  -> platinum_manifest
```

Implement CLI support for:

```text
uncertainty-flow pipeline plan --config <path>
uncertainty-flow pipeline run --config <path>
uncertainty-flow pipeline verify <run-id>
uncertainty-flow pipeline lineage <run-id>
uncertainty-flow pipeline list-runs
```

Exit gate:

- A local command produces verified Bronze through Platinum artifacts.
- `pipeline plan` produces no files and performs no data/model side effects.
- Invalid split membership fails before training.
- Repeating the same request reuses a verified run or follows an explicit rerun policy.
- Verification is required before a run can become `success`.
- DAG construction and expected output-node tests are stable.

### Phase 3 — Compatibility migration

Implement `uncertainty_flow/benchmarking/api.py` as the stable facade and route `BenchmarkFlow` through the coordinator using explicit translation functions.

Tasks:

- Map `BenchmarkConfig` to `RunRequest` while retaining legacy constructor fields.
- Map `ModelResult`/`BenchmarkResult` to Platinum metric/evidence records and back.
- Adapt existing dataset providers, model providers, tuning, and sinks behind the new interfaces.
- Route the existing `benchmark` CLI command through the compatibility facade.
- Preserve existing output names, keys, and documented CLI behavior for the first migration release.
- Add parity tests for representative benchmark runs, partial failures, and error/exit behavior.
- Add a deprecation notice only after parity and one release carrying both paths; do not remove `BenchmarkFlow` in this phase.

Exit gate:

- Existing documented CLI workflows still pass.
- Representative old/new results have equivalent core metrics and explicit field mappings.
- No module under `uncertainty_flow/core/` imports Hamilton or benchmarking.

### Phase 4 — Modular dataset/model/metric matrix

Add registries with applicability metadata for:

- Dataset adapters and source snapshot policies.
- Model providers and model artifact persistence.
- Metrics, coverage levels, horizons, and required/optional status.
- Validation plans for temporal, rolling-origin, and later sliding-window evaluation.

Add execution policy for:

- Selective outputs and cached artifact reuse.
- Independent model-branch failures.
- Required evaluation versus optional diagnostics.
- `degraded` status with node, exception category, configuration, evidence impact, and remediation.

Only after the single-model lifecycle is stable, evaluate Hamilton `@subdag` for repeated model evaluation and `@parameterized_subdag` for matrix expansion. Node naming and contract tests must remain predictable.

Exit gate:

- Adding a model, dataset, or metric does not require coordinator edits.
- One failed independent optional branch does not invalidate successful required branches.
- Required/optional status is visible in the run manifest.
- Split assignments are persisted and never reconstructed implicitly.

### Phase 5 — Astro/Starlight evidence portal

Build a separate `evidence-site/` Astro/Starlight project initially, consuming verified Platinum exports only.

Add:

- Versioned schemas for run, dataset, model, metric, and partition-index records.
- Streaming partitioned `jsonl.gz` exporter.
- Small `src/data/generated/index.json` containing latest runs, partitions, counts, checksums, and schema version.
- Build-time loader that decompresses selected partitions and generates static pages.
- Run, dataset, model, metric, calibration, and lineage pages.
- Gzip integrity, schema, record-count, and checksum tests.
- Separate GitHub Pages build/deployment workflow with explicit ownership of the new site output.

Do not copy prediction-level Parquet data into the static site. Keep the current MkDocs `site/` output untouched until a separate documentation cutover decision.

Exit gate:

- Site builds from verified Platinum artifacts only.
- Corrupt or invalid partitions fail export/build.
- Latest run pages are navigable without loading historical evidence.
- Site deployment does not change the current library documentation workflow unintentionally.

### Phase 6 — Operational hardening, only as justified

Implement only where actual operating needs exist:

- Object-store adapter behind `ArtifactStore`.
- Run-scoped locks and stale-run recovery.
- Retention/garbage collection that protects published lineage.
- Scheduler integration and structured node events.
- Alerts and dashboards.
- Signed or externally verifiable release manifests.

Exit gate:

- Concurrent runs cannot publish conflicting artifacts.
- Recovery procedures are tested.
- Retention never deletes artifacts referenced by published runs.

## Reviewable PR sequence

1. Baseline fixtures, ADRs, and compatibility snapshots.
2. Contracts, canonical identities, and local artifact-store protocol.
3. Local storage implementation, medallion layout, atomic writes, and checksums.
4. Hamilton driver, vertical-slice nodes, coordinator, dry-run, and verifier.
5. `BenchmarkFlow`/`BenchmarkRunner` compatibility facade and CLI routing.
6. Registries, temporal/rolling validation, selective execution, and branch policy.
7. Platinum evidence exporter and versioned `jsonl.gz` partitions.
8. Astro/Starlight evidence portal and independent deployment workflow.
9. Object storage/scheduler/locking/retention adapters only when required.

Keep the core pipeline migration and the evidence portal in separate PRs.

## Test and CI gates

Every phase should add tests at the same boundary it introduces:

- Unit: contracts, canonical hashes, adapters, storage paths, serializers, and verification checks.
- DAG: node set, dependency construction, configuration variants, selected outputs, and dry-run side-effect absence.
- Integration: local Bronze-to-Platinum run, idempotent rerun, corruption, required failure, optional degradation, and compatibility parity.
- Properties: split disjointness/exhaustiveness, temporal ordering, interval ordering, identifier-aligned outputs, and identity stability.
- Site: schema validation, gzip integrity, partition checksums, and build from verified artifacts only.

Required commands for the Python pipeline:

```bash
uv run ruff check uncertainty_flow/ tests/
uv run ruff format uncertainty_flow/ tests/ --check
uv run mypy uncertainty_flow/
uv run pytest tests/ -q
```

Default CI must remain network-free. Network-backed dataset checks stay separately marked. Add Hamilton and site-specific gates only when the relevant phase lands.

## Initial configuration

Start with the plan’s shape, but resolve it into an immutable runtime configuration before DAG construction:

```yaml
pipeline:
  mode: benchmark
  reuse_policy: reuse_verified
  fail_fast: false
dataset:
  provider: local_parquet
  id: example_regression
  uri: data/input/example.parquet
  target: y
validation:
  strategy: random_holdout
  test_size: 0.2
  calibration_size: 0.2
  random_seed: 42
models:
  - id: conformal_gbr
    provider: conformal_regressor
    parameters: {}
evaluation:
  coverage_levels: [0.8, 0.9, 0.95]
  metrics: [coverage, sharpness, winkler, pinball]
  diagnostics:
    feature_leverage: optional
    shap: disabled
storage:
  provider: local
  root: data
publication:
  platinum: true
  site_evidence: false
```

Configuration precedence must be documented and tested as: CLI override, environment, file, application defaults.

## Open decisions to resolve in Phase 0

- Exact Hamilton dependency/version and whether it belongs in core dependencies or a benchmarking optional extra.
- Pydantic versus frozen dataclasses for each contract family; use one convention unless a documented boundary requires otherwise.
- Whether Bronze snapshots are always copied locally or may reference immutable external content.
- The first production dataset fixture and its license metadata.
- Whether calibration is a required first-slice artifact for every model or only for models claiming calibrated intervals.
- Astro/Starlight deployment path and eventual relationship to the existing MkDocs GitHub Pages site.
- Whether `.uf` model archives are sufficient for Platinum model artifacts or need a signed manifest before operational use.

## Definition of done

The full project is complete when:

- Core model APIs remain unchanged and independent of pipeline concerns.
- Supported benchmarking executes through Hamilton with compatibility coverage.
- Every run has stable identity, resolved configuration, lineage, checksums, and a verified manifest.
- Bronze through Platinum contracts and persisted split membership are enforced.
- Required/optional failure semantics, dry-run, idempotent reuse, and recovery are tested.
- Existing benchmark workflows remain supported or have an explicit release-scoped deprecation.
- The evidence portal consumes a small index and partitioned compressed records without duplicating prediction-level Parquet.
- Documentation and CI reflect the executable contracts.
- The old monolithic path is removed or formally deprecated only after parity is demonstrated.

## Recommended next implementation task

The core local implementation and provider-neutral local operational hardening are complete, including fold-level rolling-origin execution and aggregation. Remaining operational work is selecting and implementing a concrete object-store adapter, scheduler integration, distributed locking beyond one filesystem, alert delivery, retention policy for published history, and an externally managed signing service. The deployment integrations require a target platform and credential model that are not specified by this repository plan. Continue from `docs/plans/20260715-external-integrations-and-debt-handoff.md`, which separates those decision-dependent integrations from bounded local debt.

## 2026-07-13 implementation review and completion plan

### Implementation progress

- 2026-07-13: Phase A partial completion: `pipeline verify`, `pipeline lineage`, and `pipeline list-runs` now operate on persisted manifests; configuration resolution follows CLI override, environment, file, then defaults for storage root and reuse policy.
- 2026-07-13: Matrix execution now verifies each materialized artifact before a run can verify, writes the resolved configuration as a checksummed artifact, re-raises process-control exceptions, and requires failed model branches to be explicitly marked optional.
- 2026-07-13: B0 completed: the supported model, dataset, validation, tuning, metric, command, and output inventory is frozen below. All currently documented names have an explicit pipeline disposition before provider migration begins.
- 2026-07-14: B1 completed: single-model and matrix execution now return the same immutable `PipelineRunResult` with typed `ModelExecutionResult` records, including resolved parameters, timing, row counts, metrics, branch status, and artifact refs; flat migration-only result views were removed at the breaking cutover.
- 2026-07-14: B2/B3 complete for the retained local execution surface: all thirteen inventory model names resolve through typed provider adapters, provider-specific defaults/rejection are enforced, metrics are executable registry entries, local Parquet and pinned Hugging Face adapter/version metadata participate in identity, and providers with native `.uf` save support publish verified model artifacts. Richer applicability predicates and model-specific tuning schemas remain open.
- 2026-07-13: B4 substantially complete for local publication: execution writes to an isolated staging store, verifies checksummed artifacts, signs manifests when publication credentials are supplied, promotes the final manifest last, and reuse validates identity, status, artifact checksums, and configured authenticity. External signing and multi-filesystem/object-store atomicity remain open.
- 2026-07-14: B5 completed: package `benchmark` and `tune` resolve datasets through registered adapters and use typed provider/tuning paths; the redundant standalone `benchmarks/run_benchmarks.py` program was removed after provider-registry coverage migrated.
- 2026-07-14: B6 completed: legacy tests and compatibility exports were removed, the benchmark guide/report/docs use the canonical CLI, the notebook source and regenerated freeze output are pipeline-native, and rolling-origin fold aggregation is covered.
- 2026-07-14: B7 completed: the standalone benchmark program and compatibility-only modules were deleted, package exports were cut over, and legacy-only tests were removed.
- 2026-07-14 final review completed: resolved configuration now redacts signing secrets, uses normalized registry specs, and persists the exact code version used by identity; reuse reconstructs exact typed results and verifies configured authenticity; duplicate provider variants remain distinct; datetime inputs hash canonically; CLI output flags and lifecycle errors are correct; tuning honors sample limits and cooperative timeouts; evidence/report readers consume only the pipeline-native schema with manifest dataset metadata.
- 2026-07-14 final review also corrected publication edge cases: empty matrices fail before execution, `fail_fast` cannot publish a silently truncated matrix, rolling-origin aggregates no longer expose the last fold model as a representative fitted artifact, failed verification never promotes a manifest, stale staging is cleared before rebuild, and GC removes abandoned staging while preserving active locked runs.
- Local release gates pass: the final full coverage run completed with `1110 passed, 45 skipped` at `85.62%`; Ruff, package-wide mypy, touched-surface formatting, non-strict MkDocs, evidence-site build, exhaustive live legacy search, and `git diff --check` pass. Repository-wide format checking still reports the untouched pre-existing `scripts/report_touched_modules.py`; strict MkDocs still reports pre-existing API annotation/archive-link warnings.

### Review conclusion

The Hamilton/medallion path now owns the package `benchmark`, `tune`, local dataset loading, thirteen registered provider names, staged publication, rolling-origin fold aggregation, and typed results. The standalone benchmark program, compatibility modules, legacy tests, and stale documentation/notebook output have been removed, and the final local release gate passes.

The completion work deliberately changes the project policy from temporary compatibility to a clean pipeline-only public surface. This is a breaking release: do not preserve aliases, deprecation shims, or legacy JSON/CSV field contracts after the cutover.

### Confirmed gaps and concerns

| Priority | Status | Finding | Evidence | Required remediation |
| --- | --- | --- | --- | --- |
| P0 | Resolved | The package benchmark command depended on the legacy runner path. | `cli.py::benchmark` now constructs `RunRequest` and executes `ModelMatrixCoordinator`; no live import of `BenchmarkRunner` or `BenchmarkFlow` remains in that command. | Preserve the import-tracing CLI test while migrating the remaining standalone path. |
| P0 | Resolved | The legacy translator was not behaviorally equivalent. | The translator API and its compatibility tests were deleted; callers use `RunRequest`, typed provider results, and verified pipeline artifacts directly. | Use the pipeline-native request/result contracts. |
| P0 | Resolved | The standalone benchmark program was a second live legacy runtime. | `benchmarks/run_benchmarks.py` and its wrapper tests were deleted after all thirteen retained providers moved into the typed registry/matrix path. | Use the canonical package CLI or coordinators. |
| P0 | Resolved | The new public result contract was too weak for a pipeline-only cutover. | `PipelineRunResult` contains immutable per-model identity, status, timing, row count, metrics, and artifact references; callers use `model_results` directly. | Use the typed per-model records. |
| P1 | Resolved | Metric and model configuration were only partially contract-driven. | All thirteen retained names resolve through typed registry providers; provider-specific defaults/rejection, executable metrics, resolved parameters, train time, row counts, prediction artifacts, and verified native `.uf` model artifacts where supported are covered. | Add richer applicability/tuning schemas when the provider contract is expanded. |
| P1 | Resolved | Dataset execution bypassed the dataset registry in the canonical CLI. | `pipeline run`, `benchmark`, and `tune` load through registered adapters; the default registry now exposes local Parquet and pinned `hf://<dataset>@<revision>` HuggingFace loading, with adapter version/revision inputs in request identity. | Add richer schema/applicability validation when the remote dataset contract is expanded. |
| P1 | Resolved | Legacy references extended beyond the Phase B documentation list. | The notebook source and regenerated `_freeze`/HTML output, guides, contributing page, changelog, and API pages now use pipeline contracts; stale compatibility-only tests and modules were deleted. | Keep generated notebook output synchronized through the documented Quarto workflow. |
| P1 | Resolved | Artifact verification had a publication-order gap. | Runs write through an isolated staging store, verify each materialization and verification report, optionally sign the manifest, and promote the final manifest last; reuse validates checksums and configured HMAC authenticity. | Add object-store promotion and externally managed signing when deployment targets are selected. |
| P2 | Resolved | Medallion layer semantics were underspecified. | Bronze now records source-faithful schema metadata, Silver adds/canonicalizes stable string row identity with a normalization report, and Gold records model-ready split/fold membership; coordinator tests assert the schema/content transition. | Preserve these layer contracts as adapters evolve. |
| P2 | Resolved | The release verification statement needed to distinguish targeted and full gates. | The full coverage gate passes at `1085 passed, 45 skipped` and `85.60%` against the 80% threshold; the post-persistence full non-coverage suite passes at `1087 passed, 45 skipped`, with focused post-patch static/diff checks passing. | Keep both commands in the handoff. |
| P1 | Resolved | The requested operational CLI lifecycle was incomplete. | `pipeline verify`, `pipeline lineage`, and `pipeline list-runs` are now registered and covered by `test_pipeline_operations_cli.py`. | Keep the lifecycle tests and add exit-code assertions for corrupt and missing runs. |
| P1 | Resolved | Matrix materializations and process-control exceptions were unsafe. | Matrix execution now records materialization verification, re-raises `KeyboardInterrupt`/`SystemExit`, and only degrades explicitly optional model branches. | Preserve these invariants in the pipeline-native result and publication tests. |
| P1 | Resolved | Configuration precedence was absent. | CLI override, environment, file, and default precedence is now implemented and tested for storage root and reuse policy. | Extend typed resolution and precedence tests to every supported override, not only storage/reuse. |

### Blocker execution plan

Implement the work packages in order. Do not start legacy deletion until B0-B6 are complete and the Phase B full gate passes.

#### B0 — freeze the supported behavior inventory

**Purpose:** prevent the cutover from silently dropping behavior that currently lives outside the package CLI.

- Inventory model names, dataset modes, validation strategies, tuning options, output fields, and command options from `cli.py`, `BenchmarkFlow`, `BenchmarkRunner`, `benchmarks/run_benchmarks.py`, tests, docs, and notebooks.
- Classify each item as `retain in pipeline`, `replace with named pipeline behavior`, or `remove as a documented breaking change`.
- Default to retaining the three package CLI models and every additional standalone model that is asserted by tests or documented as supported. Removal requires an explicit entry in the breaking-change table.
- Record the final inventory in this plan before provider work begins; use it as the source for registry and integration-test parameterization.

**Deliverable:** a checked model/behavior matrix in this document with one disposition and replacement path per item.

**Gate:** no supported item is left with an undecided disposition.

**B0 inventory (frozen 2026-07-13).** The inventory was reconciled against `uncertainty_flow/cli.py`, `benchmarking/flow.py`, `benchmarking/runner.py`, `benchmarking/configs.py`, `benchmarking/tuning.py`, `benchmarks/run_benchmarks.py`, `tests/benchmarking/`, `tests/test_cli.py`, `docs/api/`, `docs/guides/benchmarking.md`, and `docs/benchmarks/README.md`.

**Model matrix.** Every model currently exposed by the package CLI or documented/asserted by the standalone benchmark surface is retained. Optional dependencies are provider capabilities and must produce an explicit optional-branch degradation, not an import-time registry failure.

| Model name | Evidence | Disposition | Pipeline replacement and required behavior |
| --- | --- | --- | --- |
| `quantile-forest` | Package CLI, tuning search space, tests, benchmark docs | Retain in pipeline | Registry provider; preserve `horizon`, `n_estimators`, `random_state`, calibration, and tuning parameters; time-series applicability. |
| `conformal-regressor` | Package CLI, compatibility tests, benchmark docs | Retain in pipeline | Registry provider; preserve base-estimator, calibration, seed, and typed tuning parameters; tabular applicability. |
| `conformal-forecaster` | Package CLI, tuning search space, tests, benchmark docs | Retain in pipeline | Registry provider; preserve `horizon`, `lags`, estimator, calibration, seed, and tuning parameters; time-series applicability. |
| `deep-quantile` | Standalone `UF_MODELS`, benchmark outputs/docs | Retain in pipeline | Optional registry provider backed by `DeepQuantileNet`; declare dependency and serializability capabilities. |
| `deep-quantile-torch` | Standalone `UF_MODELS`, benchmark outputs/docs | Retain in pipeline | Optional registry provider backed by `DeepQuantileNetTorch`; ML extra is an explicit applicability/degradation condition. |
| `transformer-forecaster` | Standalone `UF_MODELS`, benchmark docs (currently skipped without Chronos) | Retain in pipeline | Optional registry provider backed by `TransformerForecaster`; preserve horizon/model parameters and report missing Chronos capability as an optional failure. |
| `bayesian-quantile` | Standalone `UF_MODELS`, benchmark outputs/docs | Retain in pipeline | Optional registry provider backed by `BayesianQuantileRegressor`; declare NumPyro capability, parameters, and persistence limits. |
| `linear-regression` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry baseline provider using conformalized OLS behavior; preserve calibration and seed settings. |
| `ridge-regression` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry baseline provider; validate and persist `alpha`, calibration, and seed. |
| `random-forest` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry baseline provider; validate and persist estimator, leaf, calibration, and seed settings. |
| `gradient-boosting` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry baseline provider; preserve its named behavior even where current defaults match conformal regression, until a separate breaking-change decision removes it. |
| `naive-forecast` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry deterministic baseline; preserve horizon and residual-interval semantics and record zero/near-zero training time honestly. |
| `moving-average` | Standalone `BASELINE_MODELS`, benchmark outputs/docs | Retain in pipeline | Registry deterministic baseline; validate and persist window/horizon settings and record its capability limits. |

**Dataset and execution modes.** These are all retained through named registry adapters; the CLI must not contain format-specific readers.

| Existing mode | Disposition | Pipeline replacement |
| --- | --- | --- |
| Named datasets in `AVAILABLE_DATASETS` (including local fixtures such as `concrete`, `wine_quality`, `energy_efficiency`, and synthetic datasets) | Retain in pipeline | Resolve metadata through `DatasetRegistry`, load through a provider/versioned adapter, validate target/schema, and include the resolved contract in identity. |
| Named HuggingFace/Chronos datasets and arbitrary `path/subset` HuggingFace references | Retain in pipeline | Add an explicit network-capable HuggingFace adapter; revision pinning is part of the dataset contract, while local Parquet remains the mandatory network-free integration path. |
| Local Parquet URI/path | Retain in pipeline | Use the registered `local_parquet` adapter; direct `pl.read_parquet` is confined to that adapter. |
| Standalone default datasets `weather`, `exchange_rate`, `electricity` | Retain in pipeline | Preserve as the default multi-run selection for a pipeline-native standalone benchmark entry point. |
| Standalone dataset groups `ts`, `tabular`, `synthetic`, and `expanded`, plus `--all-datasets` | Retain in pipeline | Resolve groups to typed batches of `RunRequest` values and execute each run through the same coordinator; no second model/runtime path. |

**Validation, tuning, and failure behavior.**

| Existing behavior | Disposition | Pipeline replacement |
| --- | --- | --- |
| Random holdout with `test_size`, random seed, and calibration membership | Retain in pipeline | Persist deterministic train/calibration/test assignments and leakage evidence in the validation contract. |
| Temporal holdout for time-series models | Retain in pipeline | Persist ordered cutoffs and memberships; provider applicability selects this strategy. |
| Hybrid validation (outer holdout plus inner out-of-sample CV) | Retain in pipeline | Make tuning a typed pre-fit operation with persisted inner memberships, trial inputs, result, and selected parameters. |
| Rolling-origin evaluation (`rolling_origin`, split count, minimum train size, rolling horizon) | Retain in pipeline | Implement real fold-level execution, persisted fold memberships, and aggregate fold metrics; no tail-holdout fallback or legacy delegation. |
| Auto-tuning on/off, target coverage, tuning sample limit, timeout, search spaces, and dataset revision | Retain in pipeline | Registry/model schemas own parameter validation; tuning outputs are pipeline artifacts and resolved settings, not legacy `TuningResult` translation. |
| `allow_partial` model continuation | Replace with named pipeline behavior | Express branch policy using typed `required` model specs plus `fail_fast`; required failure fails the run, optional failure degrades it. |

**Metrics and result fields.** The canonical evaluator retains the currently emitted probabilistic and point metrics, with metric schemas declaring capability and required/optional policy: `coverage`, `sharpness`, `winkler`, `pinball`, `crps`, `mae`, `rmse`, and `calibration_error`. It also retains evaluation levels (currently 0.8 and 0.9), per-model train time, evaluation row count, validation strategy/fold metadata, resolved parameters, and artifact references. Legacy flattened names such as `coverage_90`, `winkler_80`, `pinball_loss`, and `train_time_sec` are presentation mappings only; they are not fields of the pipeline-native result contract.

**Commands and output behavior.**

| Existing surface | Disposition | Pipeline replacement |
| --- | --- | --- |
| `benchmark` | Retain as a pipeline-native compatibility-period command | Construct a canonical `RunRequest`, execute the dataset registry and matrix coordinator, and render typed results; it must not import or instantiate `BenchmarkRunner`/`BenchmarkFlow`. |
| `tune` | Retain as a pipeline-native tuning command | Resolve dataset/model registries and emit typed tuning artifacts/results; no direct `auto_tune` legacy path. |
| `list-datasets`, `download-dataset`, and `download-all` | Retain in the dataset adapter surface | Read/write through dataset registry adapters and preserve revision/cache controls. |
| `pipeline plan`, `run`, `verify`, `lineage`, `list-runs`, `export-site`, and `gc` | Retain as canonical lifecycle commands | Keep manifest, evidence, verification, and safe-GC exit-code contracts. |
| `--model`, `--dataset`, `--samples`/`--n-samples`, `--horizon`, `--n-estimators`, `--target`, `--auto-tune`, `--target-coverage`, `--tune-samples`, `--test-size`, `--dataset-revision`, and `--hybrid-validation` | Retain with typed request mapping | Normalize aliases into one validated request schema and persist resolved values. |
| `--output`, `--json-only`, and `--csv-only` | Replace with named pipeline export behavior | Primary output is the verified run/evidence manifest; optional JSON/CSV exports render the typed result and never recreate the legacy result shape. |
| Legacy JSON/CSV top-level keys `results`, `errors`, `metadata`, `models`, and fabricated zero-valued fields | Remove at breaking cutover | Replace with `PipelineRunResult`, `ModelExecutionResult`, manifest refs, and explicit status/degradation records. |

**Explicit breaking-change entries required at cutover.** Remove the public legacy symbols `BenchmarkConfig`, `BenchmarkResult`, `ModelResult`, `BenchmarkRunner`, `BenchmarkFlow`, `ResultSink`, `MODEL_REGISTRY`, `register_model`, and translator functions `request_from_legacy_config`/`legacy_result_from_verified`/`run_verified`; remove their legacy field aliases and import paths after B1-B6 gates pass. The 13 model names and the dataset/validation/tuning behaviors above are not breaking removals.

#### B1 — define the pipeline-native public contracts

**Depends on:** B0.

**Primary files:** `contracts/runs.py`, a new or existing result-contract module under `contracts/`, `coordinator.py`, `matrix.py`, `benchmarking/__init__.py`.

- Add immutable `ModelExecutionResult` and `PipelineRunResult` contracts containing model/provider identity, status, resolved parameters, train time, evaluation row count, metrics, errors/degradation reason, and model/prediction/metric artifact refs.
- Replace the coordinator's flat metric dictionary and the matrix's separate metrics/error dictionaries with the same typed result contract.
- Keep run status derived from required/optional model outcomes plus artifact verification; never infer success merely because one model completed.
- Export only pipeline-native request/result/manifest types from the eventual public API.

**Tests:** contract immutability and validation, JSON round-trip, single-model/matrix parity, required failure, optional degradation, and empty-success rejection.

**Gate:** CLI and evidence code can render a result without importing `results.py`, `sinks.py`, or `api.py`.

#### B2 — make registries executable and configuration typed

**Depends on:** B0-B1.

**Primary files:** `registry.py`, `providers.py`, `model_contracts.py`, `matrix.py`, `dataflows/modeling.py`, `tuning.py`.

- Give every retained model provider a typed parameter schema, applicability predicate, builder, serializer, and declared artifact requirements.
- Remove the matrix's `horizon=1`; resolve horizon, estimator count, seed, calibration, lags, tuning parameters, and model-specific options from the validated request.
- Give every metric registration an evaluator, parameter schema, required/optional policy, and prediction-capability requirements.
- Integrate tuning as an explicit pre-fit pipeline operation with persisted tuning inputs/results; do not call legacy flow helpers.
- Persist fitted model artifacts when the model supports safe persistence. If a model cannot be serialized, record that capability explicitly and do not fabricate a model ref.

**Tests:** parameter rejection, applicability rejection before writes, every retained provider, every retained metric, tuning enabled/disabled, and resolved parameter persistence.

**Gate:** no model/metric branching in the coordinator or CLI by string name; registry metadata drives execution.

#### B3 — route datasets and identities through one canonical adapter path

**Depends on:** B1.

**Primary files:** `registry.py`, `datasets.py`, `cli.py`, `identity.py`, `lineage.py`, request/config contracts.

- Replace the direct `pl.read_parquet()` call in `pipeline run` with `DatasetRegistry.load()`.
- Validate provider, URI, target, schema, and validation applicability before constructing execution artifacts.
- Resolve and persist dataset adapter name/version, source revision, selected target, row limit/sampling policy, and source checksum.
- Include the exact resolved dataset contract in dataset and run identity inputs.
- Keep network-free local Parquet as the mandatory integration path; make remote dataset fetching an explicit adapter capability.

**Tests:** unknown provider, unreadable URI, missing target, schema mismatch, adapter-version identity change, deterministic sampling, and local Parquet end-to-end execution.

**Gate:** canonical CLI execution has no dataset-format-specific read logic.

#### B4 — close publication and medallion-contract gaps

**Depends on:** B1-B3.

**Primary files:** `storage/base.py`, `storage/local.py`, `coordinator.py`, `matrix.py`, `lineage.py`, artifact/verification contracts.

- Publish into a staging run directory and verify every Bronze, Silver, Gold, and Platinum ref, including resolved config, fitted models, predictions, metrics, verification report, and staged manifest.
- Atomically promote the staged manifest to the reusable final manifest only after all required checks pass. Failed or interrupted staging directories must not satisfy reuse.
- Make reuse validate manifest authenticity, identity, status, and every referenced checksum.
- Define medallion semantics explicitly: Bronze is source-faithful ingestion plus source metadata; Silver is normalized typed data plus row identity; Gold is validation membership and model-ready features; Platinum is run evidence.
- If Bronze and Silver content is intentionally identical for a pass-through adapter, record distinct contracts and transformation metadata rather than claiming an unperformed cleaning step.

**Tests:** corrupt write, corrupt staged/final manifest, interruption before promotion, failed-run non-reuse, missing artifact, HMAC failure, idempotent verified reuse, and layer schema/content assertions.

**Gate:** a success/degraded final manifest is the last write and proves all referenced artifacts were verified.

#### B5 — migrate all live entry points

**Depends on:** B1-B4.

**Primary files:** `cli.py`, `benchmarking/__init__.py`, `benchmarks/run_benchmarks.py`, CLI and integration tests.

- Route the top-level `benchmark` UX to the canonical request resolver, dataset registry, and matrix coordinator, or remove it in favor of `pipeline run` as an explicit breaking CLI change.
- Preserve only command options represented by typed request fields; reject removed options with normal Click usage errors during development, then remove them at cutover.
- Replace standalone class registration with pipeline providers. If B0 classifies `benchmarks/run_benchmarks.py` as redundant, delete it only after equivalent retained-model coverage exists through the canonical CLI/API.
- Convert output rendering to the typed pipeline result and artifact locations. Do not recreate legacy JSON/CSV fields with zero or invented values.

**Tests:** CLI help, invalid model/config, all retained model selections, partial optional failure, output paths, exit codes, and proof that no command imports or instantiates `BenchmarkRunner`/`BenchmarkFlow`.

**Gate:** runtime import tracing and `rg` show no live entry point depending on compatibility modules.

#### B6 — migrate documentation, notebooks, and tests

**Depends on:** B5.

- Rewrite API, CLI, guide, architecture, contributing, changelog, and notebook sources around the pipeline contracts and medallion lifecycle.
- Replace legacy snapshot/flow/runner/sink tests with contract, registry, coordinator, matrix, storage, and CLI coverage.
- Keep fixtures only when they remain valid pipeline inputs or artifact-contract examples.
- Regenerate versioned notebook freeze output if the project still treats it as generated source; otherwise remove it through the normal notebook cleanup workflow.
- Add a breaking-release migration table mapping every removed Python symbol, command option, and output shape to its replacement.

**Gate:** full repository tests and documentation checks pass, and the legacy search returns only intentional ADR/changelog history.

#### B7 — remove compatibility and legacy code

**Depends on:** B0-B6 and the Phase B full gate.

- Delete `api.py`, `flow.py`, `runner.py`, `configs.py`, `results.py`, and `sinks.py`.
- Remove class-registry code and compatibility resolution from `providers.py`, plus all legacy re-exports.
- Delete legacy-only tests, fixtures, documentation, notebook snippets, and obsolete CLI options identified by B0.
- Run the exhaustive search and import smoke tests before the final repository gate.

**Gate:** the Phase C exit criteria below pass with no aliases, shims, or deprecation wrappers retained.

**Completed 2026-07-14.** The deletion, export cutover, generated notebook refresh, legacy search, full coverage gate, package type check, docs build, and evidence-site build are complete. Remaining strict-format/docs warnings are unrelated pre-existing repository debt recorded above.

#### Execution checkpoints

1. **Checkpoint 1 — contracts:** B0-B1 complete; targeted contract/coordinator tests pass.
2. **Checkpoint 2 — authoritative engine:** B2-B4 complete; every retained provider runs through verified staging publication.
3. **Checkpoint 3 — caller cutover:** B5-B6 complete; full repository and documentation gates pass while legacy modules still exist but have no live readers.
4. **Checkpoint 4 — deletion:** B7 complete; exhaustive legacy search, full gates, and evidence-site build pass.

### Phase A — make the pipeline complete and authoritative

1. Introduce a pipeline-native public API in `uncertainty_flow/benchmarking/` built around `RunRequest`, verified run results, and typed per-model metric records. Do not expose `BenchmarkConfig`, `BenchmarkResult`, `ModelResult`, or translator functions from the new API.
2. Harden the now-complete operational CLI in `uncertainty_flow/cli.py`: retain `pipeline plan/run/verify/lineage/list-runs/export-site/gc` and define precise exit codes for invalid configuration, unverified runs, and missing run IDs.
3. Move all supported built-in providers into the registry-backed matrix path. Preserve each provider's real horizon, estimator, seed, calibration, and tuning settings in the resolved configuration and manifest. Either implement fold-level rolling-origin execution with persisted fold memberships and aggregate metrics, or reject it at request validation with no legacy fallback.
4. Make the registry operational: a model provider must declare parameter schema and applicability; a metric must declare its evaluator and required/optional status; unsupported model/dataset/validation combinations must fail before materialization.
5. Tighten publication: verify every Bronze/Silver/Gold/Platinum ref (including model, prediction, metrics, verification, and manifest references) after write and before status is `success` or `degraded`. Re-raise process-control exceptions; only declared optional branches may become `degraded`.
6. Extend the implemented CLI override, environment, file, defaults resolution to every supported field. Persist the resolved configuration as a checksummed Platinum artifact and derive identity from that exact representation.
7. Route dataset loading through the dataset registry and add the resolved adapter/version to the identity inputs; direct `pl.read_parquet` belongs inside the local adapter, not the CLI.
8. Define one supported model inventory across the package CLI and `benchmarks/run_benchmarks.py`. Port retained standalone adapters to provider registrations or explicitly remove them as part of the breaking cutover.

Acceptance gate:

```bash
uv run pytest tests/benchmarking -q --no-cov
uv run ruff check uncertainty_flow/benchmarking tests/benchmarking uncertainty_flow/cli.py
uv run ruff format uncertainty_flow/benchmarking tests/benchmarking uncertainty_flow/cli.py --check
uv run mypy uncertainty_flow/
```

Add integration tests for every built-in model, real rolling-origin folds or explicit rejection, CLI lifecycle commands, config precedence, process interrupts, corrupt materializations, and verified-run reuse.

### Phase B — migrate live callers and documentation

1. Change the top-level `benchmark` command to construct and execute the canonical pipeline request, or replace it with the `pipeline run` UX and update the documented entry point. It must no longer instantiate `BenchmarkRunner` or `BenchmarkFlow`.
2. Replace documentation in `docs/api/benchmarking.md`, `docs/api/cli.md`, `docs/api/core.md`, `docs/guides/benchmarking.md`, and `docs/architecture/overview.md` with the actual pipeline contracts, artifact paths, supported models, validation strategies, and recovery commands.
3. Replace compatibility snapshot tests with pipeline contract and CLI integration tests. Keep only fixtures that describe inputs/artifacts the supported pipeline still consumes.
4. Run the full repository gate, including the configured coverage policy, before cutover. If coverage remains unrelated historical debt, document and isolate that as a separate CI-policy change; do not report a `--no-cov` subset as full verification.
5. Update `notebooks/05_benchmarks.qmd`, `docs/project/contributing.md`, and the changelog; regenerate the notebook freeze if frozen outputs remain versioned. Search `benchmarks/`, `notebooks/`, README files, and package exports as well as `docs/` before declaring migration complete.
6. Add a pipeline-native standalone benchmark entry point only if it remains a supported product surface. Otherwise delete `benchmarks/run_benchmarks.py` and its registry-specific tests instead of leaving a second orchestration path.

Acceptance gate:

```bash
uv run ruff check uncertainty_flow/ tests/
uv run ruff format uncertainty_flow/ tests/ --check
uv run mypy uncertainty_flow/
uv run pytest tests/ -q
```

### Phase C — breaking removal of compatibility and legacy layers

Only begin after Phase B’s public CLI and library callers use the pipeline path and the full gate passes.

1. Delete `uncertainty_flow/benchmarking/api.py`, `flow.py`, `runner.py`, `configs.py`, `results.py`, `sinks.py`, and the class-registry compatibility code in `providers.py`.
2. Remove `BenchmarkFlow`, `BenchmarkRunner`, `BenchmarkConfig`, `BenchmarkResult`, `ModelResult`, `ResultSink`, `MODEL_REGISTRY`, `register_model`, `_LegacyConfig`, and all compatibility re-exports from `benchmarking/__init__.py`.
3. Delete legacy-only tests and baseline JSON/CLI fixtures; remove their imports and references from docs, changelog, ADRs, and package exports.
4. Remove obsolete command options whose semantics cannot be represented by the resolved pipeline request. Replace them with explicit pipeline configuration fields rather than hidden translation defaults.
5. Add one release-note entry labelled breaking change that names the deleted APIs and gives the pipeline replacement. Do not retain runtime aliases or warnings.
6. Remove legacy registrations and imports from `benchmarks/run_benchmarks.py`, or delete that program if Phase B retires it. Remove stale generated notebook output only through the repository's documented regeneration policy.

Exit criteria:

- `rg -n "BenchmarkFlow|BenchmarkRunner|BenchmarkConfig|BenchmarkResult|ModelResult|ResultSink|MODEL_REGISTRY|register_model|_LegacyConfig|legacy|compatibility" uncertainty_flow/benchmarking uncertainty_flow/cli.py benchmarks tests/benchmarking docs notebooks` returns only intentional historical ADR/changelog references.
- The default benchmark execution path produces verified medallion artifacts and has no import or runtime dependency on deleted modules.
- The full lint, format, type-check, and pytest gate passes.
