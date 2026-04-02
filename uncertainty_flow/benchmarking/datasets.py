"""Dataset registry and loading utilities for benchmarking."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

DATASETS_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    name: str
    hf_path: str
    subset: str | None
    domain: str
    description: str
    default_target: str
    is_local: bool = False


AVAILABLE_DATASETS: dict[str, DatasetInfo] = {
    "weather": DatasetInfo(
        name="weather",
        hf_path="ts-arena/weather",
        subset=None,
        domain="Climate",
        description="Weather time series data",
        default_target="OT",
    ),
    "exchange_rate": DatasetInfo(
        name="exchange_rate",
        hf_path="ts-arena/exchange_rate",
        subset=None,
        domain="Finance",
        description="Daily exchange rates",
        default_target="OT",
    ),
    "electricity": DatasetInfo(
        name="electricity",
        hf_path="lalababa/Time-Series-Library",
        subset="electricity",
        domain="Energy",
        description="Electricity demand time series",
        default_target="OT",
    ),
    "m4_daily": DatasetInfo(
        name="m4_daily",
        hf_path="autogluon/chronos_datasets",
        subset="m4_daily",
        domain="Mixed",
        description="M4 daily forecasting competition",
        default_target="OT",
    ),
    "m4_hourly": DatasetInfo(
        name="m4_hourly",
        hf_path="autogluon/chronos_datasets",
        subset="m4_hourly",
        domain="Mixed",
        description="M4 hourly forecasting competition",
        default_target="OT",
    ),
    "m4_weekly": DatasetInfo(
        name="m4_weekly",
        hf_path="autogluon/chronos_datasets",
        subset="m4_weekly",
        domain="Mixed",
        description="M4 weekly forecasting competition",
        default_target="OT",
    ),
    "m4_monthly": DatasetInfo(
        name="m4_monthly",
        hf_path="autogluon/chronos_datasets",
        subset="m4_monthly",
        domain="Mixed",
        description="M4 monthly forecasting competition",
        default_target="OT",
    ),
    "m4_quarterly": DatasetInfo(
        name="m4_quarterly",
        hf_path="autogluon/chronos_datasets",
        subset="m4_quarterly",
        domain="Mixed",
        description="M4 quarterly forecasting competition",
        default_target="OT",
    ),
    "m4_yearly": DatasetInfo(
        name="m4_yearly",
        hf_path="autogluon/chronos_datasets",
        subset="m4_yearly",
        domain="Mixed",
        description="M4 yearly forecasting competition",
        default_target="OT",
    ),
    "weatherbench_daily": DatasetInfo(
        name="weatherbench_daily",
        hf_path="autogluon/chronos_datasets",
        subset="weatherbench_daily",
        domain="Climate",
        description="WeatherBench daily weather data",
        default_target="OT",
    ),
    "weatherbench_hourly_temperature": DatasetInfo(
        name="weatherbench_hourly_temperature",
        hf_path="autogluon/chronos_datasets",
        subset="weatherbench_hourly_temperature",
        domain="Climate",
        description="WeatherBench hourly temperature",
        default_target="OT",
    ),
    "monash_electricity_hourly": DatasetInfo(
        name="monash_electricity_hourly",
        hf_path="autogluon/chronos_datasets",
        subset="monash_electricity_hourly",
        domain="Energy",
        description="Australian electricity demand",
        default_target="OT",
    ),
    "monash_london_smart_meters": DatasetInfo(
        name="monash_london_smart_meters",
        hf_path="autogluon/chronos_datasets",
        subset="monash_london_smart_meters",
        domain="Energy",
        description="London smart meter data",
        default_target="OT",
    ),
    "ercot": DatasetInfo(
        name="ercot",
        hf_path="autogluon/chronos_datasets",
        subset="ercot",
        domain="Energy",
        description="Texas electricity demand",
        default_target="OT",
    ),
    "monash_traffic": DatasetInfo(
        name="monash_traffic",
        hf_path="autogluon/chronos_datasets",
        subset="monash_traffic",
        domain="Transportation",
        description="Traffic flow data",
        default_target="OT",
    ),
    "monash_pedestrian_counts": DatasetInfo(
        name="monash_pedestrian_counts",
        hf_path="autogluon/chronos_datasets",
        subset="monash_pedestrian_counts",
        domain="Transportation",
        description="Pedestrian counts",
        default_target="OT",
    ),
    "taxi_1h": DatasetInfo(
        name="taxi_1h",
        hf_path="autogluon/chronos_datasets",
        subset="taxi_1h",
        domain="Transportation",
        description="Taxi trip counts (1h)",
        default_target="OT",
    ),
    "taxi_30min": DatasetInfo(
        name="taxi_30min",
        hf_path="autogluon/chronos_datasets",
        subset="taxi_30min",
        domain="Transportation",
        description="Taxi trip counts (30min)",
        default_target="OT",
    ),
    "monash_hospital": DatasetInfo(
        name="monash_hospital",
        hf_path="autogluon/chronos_datasets",
        subset="monash_hospital",
        domain="Healthcare",
        description="Hospital admissions",
        default_target="OT",
    ),
    "exchange_rate_kaggle": DatasetInfo(
        name="exchange_rate_kaggle",
        hf_path="autogluon/chronos_datasets",
        subset="exchange_rate",
        domain="Finance",
        description="Exchange rates from Kaggle",
        default_target="OT",
    ),
    "monash_fred_md": DatasetInfo(
        name="monash_fred_md",
        hf_path="autogluon/chronos_datasets",
        subset="monash_fred_md",
        domain="Finance",
        description="FRED macroeconomic indicators",
        default_target="OT",
    ),
    "m5": DatasetInfo(
        name="m5",
        hf_path="autogluon/chronos_datasets",
        subset="m5",
        domain="Retail",
        description="Walmart sales data",
        default_target="OT",
    ),
    "dominick": DatasetInfo(
        name="dominick",
        hf_path="autogluon/chronos_datasets",
        subset="dominick",
        domain="Retail",
        description="Retail sales data",
        default_target="OT",
    ),
    "ushcn_daily": DatasetInfo(
        name="ushcn_daily",
        hf_path="autogluon/chronos_datasets",
        subset="ushcn_daily",
        domain="Climate",
        description="US daily climate data",
        default_target="OT",
    ),
    "beijingpm25": DatasetInfo(
        name="beijingpm25",
        hf_path="autogluon/chronos_datasets",
        subset="beijingpm25",
        domain="Environment",
        description="Beijing PM2.5 air quality",
        default_target="OT",
    ),
    "kaggle_web_traffic_weekly": DatasetInfo(
        name="kaggle_web_traffic_weekly",
        hf_path="autogluon/chronos_datasets",
        subset="kaggle_web_traffic_weekly",
        domain="Web",
        description="Wikipedia web traffic",
        default_target="OT",
    ),
}


CHRONOS_DATASETS = [
    "m4_daily",
    "m4_hourly",
    "m4_weekly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
    "m5",
    "weatherbench_daily",
    "weatherbench_hourly_temperature",
    "weatherbench_hourly_pressure",
    "weatherbench_hourly_humidity",
    "weatherbench_hourly_wind_speed",
    "monash_electricity_hourly",
    "monash_london_smart_meters",
    "monash_traffic",
    "monash_pedestrian_counts",
    "monash_pedestrian_counts_2",
    "monash_car_parts",
    "monash_cif_2016",
    "monash_cif_2016_2",
    "monash_construction_120",
    "monash_construction_100",
    "monash_debuts_demand",
    "monash_debuts_demand_2",
    "monash_fred_md",
    "monash_hospital",
    "monash_hospital_2",
    "monash_kaggle_web_traffic",
    "monash_kaggle_web_traffic_2",
    "monash_kddカップ",
    "monash_kdd2022",
    "monash_london_underground",
    "monash_m1",
    "monash_m3",
    "monash_m3_extra",
    "monash_mc2_京温",
    "monash_nikkei_225",
    "monash_oikolab_etf",
    "monash_pems_l",
    "monash_pems_m",
    "monash_pems_s",
    "monash_rainfall",
    "monash_saugeen_river",
    "monash_smd",
    "monash_tourism_quarterly",
    "monash_tourism_monthly",
    "monash_tourism_weekly",
    "monash_tourism_yearly",
    "dominick",
    "ercot",
    "ett_h",
    "ett_h2",
    "ett_m1",
    "ett_m2",
    "exchange_rate",
    "exchange_rate_kaggle",
    "kaggle_web_traffic_weekly",
    "kaggle_web_traffic_daily",
    "beijingpm25",
    "temperature_rain",
    "solar_AL",
    "solar_AZ",
    "solar_CA",
    "solar_CT",
    "solar_FL",
    "solar_GA",
    "solar_IL",
    "solar_IN",
    "solar_KS",
    "solar_KY",
    "solar_MA",
    "solar_MD",
    "solar_ME",
    "solar_MI",
    "solar_MN",
    "solar_MO",
    "solar_MS",
    "solar_NC",
    "solar_NE",
    "solar_NH",
    "solar_NJ",
    "solar_NM",
    "solar_NV",
    "solar_NY",
    "solar_OH",
    "solar_OK",
    "solar_OR",
    "solar_PA",
    "solar_PR",
    "solar_SC",
    "solar_SD",
    "solar_TN",
    "solar_TX",
    "solar_UT",
    "solar_VA",
    "solar_VT",
    "solar_WA",
    "solar_WI",
    "solar_WV",
    "solar_WY",
    "sunspot",
    "taxi_1h",
    "taxi_30min",
    "temperature_domain",
    "ushcn_daily",
    "usto_utokyo_ocean",
]


for ds_name in CHRONOS_DATASETS:
    if ds_name not in AVAILABLE_DATASETS:
        AVAILABLE_DATASETS[ds_name] = DatasetInfo(
            name=ds_name,
            hf_path="autogluon/chronos_datasets",
            subset=ds_name,
            domain="Mixed",
            description=f"Chronos dataset: {ds_name}",
            default_target="OT",
        )


def list_datasets() -> list[DatasetInfo]:
    """Return all available datasets sorted by name."""
    return sorted(AVAILABLE_DATASETS.values(), key=lambda x: x.name)


def list_datasets_by_domain() -> dict[str, list[DatasetInfo]]:
    """Return datasets grouped by domain."""
    by_domain: dict[str, list[DatasetInfo]] = {}
    for ds in AVAILABLE_DATASETS.values():
        if ds.domain not in by_domain:
            by_domain[ds.domain] = []
        by_domain[ds.domain].append(ds)
    return by_domain


def get_numeric_cols(df: pl.DataFrame) -> list[str]:
    """Get only numeric columns for sklearn compatibility."""
    return [c for c in df.columns if df[c].dtype != pl.String]


def load_dataset(
    name: str,
    split: str = "train",
    n_samples: int | None = None,
    force_download: bool = False,
    cache_dir: str | None = None,
) -> tuple[pl.DataFrame, DatasetInfo]:
    """Load a dataset by name or path.

    Args:
        name: Dataset name (e.g., 'weather', 'm4_daily') or HF path
              (e.g., 'autogluon/chronos_datasets/m4_daily')
        split: Dataset split to load ('train', 'test', etc.)
        n_samples: Limit number of rows (for memory efficiency)
        force_download: Force re-download even if cached
        cache_dir: Custom cache directory

    Returns:
        Tuple of (DataFrame, DatasetInfo)

    Raises:
        ValueError: If dataset not found or cannot be loaded
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required for benchmarking. Install with: pip install datasets"
        )

    ds_info: DatasetInfo | None = None

    if "/" in name:
        parts = name.split("/")
        if len(parts) >= 3:
            hf_path = "/".join(parts[:-1])
            subset = parts[-1]
            ds_info = DatasetInfo(
                name=name,
                hf_path=hf_path,
                subset=subset,
                domain="Mixed",
                description=f"Dataset: {name}",
                default_target="OT",
            )
        else:
            ds_info = DatasetInfo(
                name=name,
                hf_path=name,
                subset=None,
                domain="Mixed",
                description=f"Dataset: {name}",
                default_target="OT",
            )
    elif name in AVAILABLE_DATASETS:
        ds_info = AVAILABLE_DATASETS[name]
    else:
        raise ValueError(
            f"Dataset '{name}' not found. "
            f"Use 'uncertainty-flow list-datasets' to see available datasets."
        )

    try:
        if ds_info.subset:
            hf_ds = hf_load_dataset(
                ds_info.hf_path,
                ds_info.subset,
                split=split,
            )
        else:
            hf_ds = hf_load_dataset(
                ds_info.hf_path,
                split=split,
            )

        arrow_table = hf_ds.data.table  # type: ignore[attr-defined]
        result = pl.from_arrow(arrow_table)
        if isinstance(result, pl.Series):
            df: pl.DataFrame = pl.DataFrame({result.name: result})
        else:
            df = result

        numeric_cols = get_numeric_cols(df)
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in dataset '{name}'")

        df = df.select(numeric_cols)

        if n_samples and n_samples < len(df):
            df = df.head(n_samples)

        return df, ds_info

    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{name}': {e}. "
            "Make sure the dataset exists on HuggingFace and you have internet access."
        ) from e


def download_dataset(name: str, cache_dir: str | None = None) -> Path:
    """Download a dataset to local cache.

    Args:
        name: Dataset name
        cache_dir: Custom cache directory

    Returns:
        Path to downloaded dataset
    """
    df, ds_info = load_dataset(name, n_samples=None, force_download=True)
    local_path = DATASETS_DIR / f"{ds_info.name}.parquet"
    DATASETS_DIR.mkdir(exist_ok=True)
    df.write_parquet(local_path)
    return local_path


def load_local_dataset(name: str, n_samples: int | None = None) -> tuple[pl.DataFrame, DatasetInfo]:
    """Load a dataset from local parquet file.

    Args:
        name: Dataset name
        n_samples: Limit number of rows

    Returns:
        Tuple of (DataFrame, DatasetInfo)

    Raises:
        ValueError: If dataset not found locally
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{name}' not in available datasets registry.")

    ds_info = AVAILABLE_DATASETS[name]
    local_path = DATASETS_DIR / f"{ds_info.name}.parquet"

    if not local_path.exists():
        raise ValueError(
            f"Local dataset '{name}' not found at {local_path}. "
            f"Use 'uncertainty-flow download-dataset {name}' to download it."
        )

    df = pl.read_parquet(local_path)

    if n_samples and n_samples < len(df):
        df = df.head(n_samples)

    return df, ds_info
