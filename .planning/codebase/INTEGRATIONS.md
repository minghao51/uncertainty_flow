# Integrations

## External APIs
- **HuggingFace Datasets API** — used via `datasets` library (optional `bench` extra) for loading benchmark time series datasets
  - No authentication required for public datasets

## Databases
- None. The project uses **local Parquet files** for caching benchmark data.

## Auth Providers
- None.

## Webhooks
- None.

## Third-Party SDKs (Optional)
| SDK | Purpose | Extra |
|-----|---------|-------|
| torch | Deep learning backend for quantile regression models | `torch` |
| chronos-forecasting | Transformer-based time series forecasting | `transformers` |
| shap | SHAP values for model explainability | `shap` |
| numpyro + jax | Bayesian inference models | `numpyro` |
| datasets | HuggingFace dataset loading for benchmarks | `bench` |

## Notes
- This is a **pure library** — no server, no database, no external service dependencies at runtime
- All optional integrations are lazy-loaded; core functionality works with just numpy/scipy/sklearn/polars
