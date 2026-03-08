"""Data loading: the original benchmark CSV and the Kaggle "Give Me Some Credit" dataset."""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BENCHMARK_PATH = DATA_DIR / "credit_risk_benchmark.csv"
GMSC_PATH = DATA_DIR / "raw" / "cs-training.csv"

# Public CC0 mirror of the Kaggle competition data, downloaded through the official Kaggle API.
GMSC_URL = "https://www.kaggle.com/api/v1/datasets/download/lihxlhx/give-me-some-credit"

TARGET = "dlq_2yrs"
RAW_FEATURES = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc",
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents",
]

# Kaggle column names -> short names used by the benchmark dataset.
GMSC_COLUMNS = {
    "SeriousDlqin2yrs": "dlq_2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "rev_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_inc",
    "NumberOfOpenCreditLinesAndLoans": "open_credit",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}


def download_gmsc(force: bool = False) -> Path:
    """Download the labelled Give Me Some Credit training file (150,000 rows)."""
    if GMSC_PATH.exists() and not force:
        return GMSC_PATH
    GMSC_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Give Me Some Credit from {GMSC_URL} ...")
    with urllib.request.urlopen(GMSC_URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    GMSC_PATH.write_bytes(archive.read("cs-training.csv"))
    print(f"Saved to {GMSC_PATH}")
    return GMSC_PATH


def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(BENCHMARK_PATH)


def load_gmsc() -> pd.DataFrame:
    df = pd.read_csv(download_gmsc(), index_col=0).rename(columns=GMSC_COLUMNS)
    df.index.name = "gmsc_id"
    return df[RAW_FEATURES + [TARGET]]


def benchmark_ids(gmsc: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Index:
    """Locate the benchmark rows inside Give Me Some Credit.

    The benchmark is an exact subsample of GMSC (all complete-case defaulters plus as many
    random non-defaulters), so every row matches. Values are rounded to absorb float noise.
    """
    cols = RAW_FEATURES + [TARGET]
    g = gmsc[cols].round(6).reset_index().drop_duplicates(subset=cols)
    matched = benchmark[cols].round(6).merge(g, on=cols, how="inner")
    return pd.Index(matched["gmsc_id"].unique(), name="gmsc_id")


if __name__ == "__main__":
    download_gmsc(force=True)
