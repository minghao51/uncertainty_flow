# Scripts

Utility scripts for `uncertainty_flow`.

## Ingestion

Download and convert sample datasets from Hugging Face:

```bash
uv run python scripts/ingest_datasets.py
```

This will:
1. Download datasets from Hugging Face
2. Convert to Parquet format (more efficient than CSV)
3. Filter to numeric columns only (sklearn-compatible)
4. Save to `data/` directory

## Trial Benchmark

Run benchmarks to test models on sample datasets:

```bash
uv run python scripts/trial_benchmark.py
```

This will:
1. Load each dataset from `data/`
2. Run each model (QuantileForestForecaster, ConformalRegressor, ConformalForecaster)
3. Measure coverage, sharpness, Winkler score, and training time
4. Output summary table to console and `benchmark_results.csv`

## Requirements

- `datasets` library (for ingestion)
- All `uncertainty_flow` dependencies
