# Core

::: uncertainty_flow.core

## Benchmarking Package Exports

Benchmark APIs are exposed from `uncertainty_flow.benchmarking` through immutable
`RunRequest`, `RunManifest`, `PipelineRunResult`, and `ModelExecutionResult`
contracts. `BenchmarkCoordinator` handles one model; `ModelMatrixCoordinator`
handles isolated multi-model branches. Registries provide executable model,
dataset, and metric adapters, while `storage/` owns checksummed publication.
