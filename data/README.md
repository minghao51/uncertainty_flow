# Sample Datasets

Sample datasets for testing and development of `uncertainty_flow`.

## Datasets

| Dataset | Rows | Columns | Size | Description |
|---------|------|---------|------|-------------|
| `weather.parquet` | 36,887 | 22 | 1.78 MB | Multivariate weather data (temperature, pressure, humidity, wind) |
| `exchange_rate.parquet` | 5,311 | 9 | 0.23 MB | Daily exchange rates for 8 currencies |
| `electricity.parquet` | 26,304 | 321 | 14.72 MB | Hourly electricity consumption |

## Usage

```python
import polars as pl
from uncertainty_flow.models import QuantileForestForecaster

# Load dataset
df = pl.read_parquet("data/weather.parquet")
target = "OT"  # observed temperature

# Train model
model = QuantileForestForecaster(
    targets=target,
    horizon=3,
    n_estimators=50,
)
model.fit(df)

# Predict
pred = model.predict(df)
print(pred.interval(0.9))  # 90% confidence interval
```

## Ingestion

Datasets are downloaded and converted via:

```bash
uv run python scripts/ingest_datasets.py
```

## Source

- `weather.parquet`, `exchange_rate.parquet`: [ts-arena](https://huggingface.co/ts-arena) datasets
- `electricity.parquet`: [lalababa/Time-Series-Library](https://huggingface.co/datasets/lalababa/Time-Series-Library)
