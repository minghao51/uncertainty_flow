# Benchmarking

## Public Exports

`uncertainty_flow.benchmarking` exports the pipeline-native contracts and coordinators:

- `RunRequest`, `RunManifest`, `PipelineRunResult`, and `ModelExecutionResult`
- `BenchmarkCoordinator` and `ModelMatrixCoordinator`
- `ModelProviderRegistry`, `DatasetRegistry`, and `MetricRegistry`
- `default_dataset_registry()` includes local Parquet and pinned HuggingFace adapters;
  remote URIs use `hf://<dataset>@<revision>`.
- `AVAILABLE_DATASETS`
- `CHRONOS_DATASETS`
- `list_datasets`
- `list_datasets_by_domain`
- `load_dataset`
- `download_dataset`
- `TuningResult`
- `auto_tune`

## Module Structure

- `contracts/`: immutable request, manifest, verification, and result contracts
- `coordinator.py` / `matrix.py`: verified single- and multi-model execution
- `registry.py`: executable model, dataset, and metric registries
- `storage/`: checksummed local artifact persistence and staged promotion
- `lineage.py`: Bronze, Silver, and Gold artifact materialization

## Stable Model-Name Contract

Built-in benchmark names remain stable:

All thirteen retained names resolve through the provider registry:

`quantile-forest`, `conformal-regressor`, `conformal-forecaster`,
`deep-quantile`, `deep-quantile-torch`, `transformer-forecaster`,
`bayesian-quantile`, `linear-regression`, `ridge-regression`, `random-forest`,
`gradient-boosting`, `naive-forecast`, and `moving-average`.

## Lifecycle

Pipeline lifecycle:

1. Resolve and validate a typed `RunRequest`.
2. Load through a registered dataset adapter and derive content identity.
3. Materialize source-faithful Bronze, normalized Silver, and split-aware Gold lineage in a staging run directory.
4. Fit registered providers, evaluate registered metrics, and write Platinum evidence.
5. Verify all checksummed artifacts, optionally sign the manifest, then promote it last.
