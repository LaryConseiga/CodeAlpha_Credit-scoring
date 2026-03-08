"""Cleaning and feature engineering, shared by training, prediction and the demo app."""

import numpy as np
import pandas as pd

from src.data import RAW_FEATURES

LATE_COLS = ["late_30_59", "late_60_89", "late_90"]

# 96 and 98 in the late-payment counters are special codes in the source data, not real counts.
LATE_SENTINELS = (96, 98)
# Revolving utilisation is a ratio; values above this are data-entry errors.
MAX_REV_UTIL = 10.0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn invalid values into NaN so they are handled by imputation (or natively by boosting)."""
    df = df[RAW_FEATURES].astype(float).copy()
    for col in LATE_COLS:
        df.loc[df[col].isin(LATE_SENTINELS), col] = np.nan
    df.loc[df["age"] < 18, "age"] = np.nan
    df.loc[df["rev_util"] > MAX_REV_UTIL, "rev_util"] = np.nan
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw inputs and add engineered features."""
    df = clean(df)
    df["total_late_payments"] = df[LATE_COLS].sum(axis=1, min_count=3)
    df["late_severity_score"] = df["late_30_59"] + 2 * df["late_60_89"] + 3 * df["late_90"]
    df["income_per_dependent"] = df["monthly_inc"] / (df["dependents"].fillna(0) + 1)
    # When income is known, debt_ratio * income gives the monthly debt payments.
    df["monthly_debt"] = df["debt_ratio"] * df["monthly_inc"]
    df["income_missing"] = df["monthly_inc"].isna().astype(float)
    return df


def signed_log1p(x):
    """Compress heavy right tails (income, debt ratio...) before scaling for linear models / NN."""
    return np.sign(x) * np.log1p(np.abs(x))


FEATURE_NAMES = list(build_features(pd.DataFrame([dict.fromkeys(RAW_FEATURES, 0.0)])).columns)
