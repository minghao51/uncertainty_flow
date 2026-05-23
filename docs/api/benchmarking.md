# Benchmarking

## Public Exports

`uncertainty_flow.benchmarking` exports:

- `BenchmarkConfig`
- `BenchmarkResult`
- `BenchmarkRunner`
- `AVAILABLE_DATASETS`
- `CHRONOS_DATASETS`
- `list_datasets`
- `list_datasets_by_domain`
- `load_dataset`
- `download_dataset`
- `TuningResult`
- `auto_tune`

## Module Structure

- `flow.py`: `BenchmarkFlow` orchestration module
- `providers.py`: provider interface seam and stable built-in model providers
- `configs.py`: benchmark/build configuration contracts
- `results.py`: benchmark result contracts
- `sinks.py`: output serialization policy via `ResultSink`
- `runner.py`: public compatibility adapter over flow/providers/sinks

## Stable Model-Name Contract

Built-in benchmark names remain stable:

- `quantile-forest`
- `conformal-regressor`
- `conformal-forecaster`

## Lifecycle

`BenchmarkFlow` lifecycle:

1. `load`
2. `split`
3. `tune-per-run-context`
4. `fit/predict`
5. `evaluate`
6. `sink`
