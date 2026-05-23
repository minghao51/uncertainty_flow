# CLI

## Benchmark Command Architecture

The `benchmark` CLI command delegates orchestration to `BenchmarkFlow` through the `BenchmarkRunner` adapter. Runtime output policy is owned by the `ResultSink` seam, so JSON/CSV serialization is no longer a runner concern.

Current benchmark JSON shape follows `ResultSink.to_dict()`:

- top-level: `dataset`, `metadata`, `errors`, `results`
- no top-level `models` alias in serialized CLI output

::: uncertainty_flow.cli
