# External Integrations

## Data Sources
| Source | Integration | Purpose |
|--------|-------------|---------|
| HuggingFace Hub | `datasets` library | Benchmark dataset loading |
| ts-arena | weather, exchange_rate | Climate & finance time series |
| autogluon/chronos_datasets | 67+ datasets | Standardized benchmarking |
| lalababa/Time-Series-Library | electricity | Additional time series |

## No External Services
- No cloud APIs (AWS, GCP, Azure)
- No database integrations (Postgres, Redis)
- No auth providers (Auth0, Firebase)
- No webhooks
- No MLOps platforms (MLflow, W&B)

## Local Data
- Parquet files in `data/` directory for offline datasets
- JSON result files in `results/` directory

## Optional Visualization
- Streamlit dashboard for exploratory analysis (lazy import, optional)
