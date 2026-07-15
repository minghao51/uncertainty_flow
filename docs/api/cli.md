# CLI

## Benchmark Command Architecture

The `benchmark` CLI command resolves a pipeline-native `RunRequest`, loads the
dataset through a registered adapter, executes `ModelMatrixCoordinator`, and
renders the immutable pipeline result. `pipeline run` exposes the same verified
medallion lifecycle for local Parquet inputs.

Current benchmark JSON shape follows `PipelineRunResult.model_dump_json()`:

- top-level: `manifest`, `verification`, `artifacts`, and `model_results`
- each `model_results` item contains provider, status, resolved parameters,
  timing, row count, metrics, and artifact references

::: uncertainty_flow.cli
