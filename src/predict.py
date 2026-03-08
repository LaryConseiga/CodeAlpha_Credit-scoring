"""Score new applicants from a CSV file.

The CSV must contain the 10 raw input columns, either with the short names used in this
project (rev_util, age, ...) or with the original Kaggle names (RevolvingUtilizationOfUnsecuredLines, ...).

Usage:
    python -m src.predict data/credit_risk_benchmark.csv -o predictions.csv
    python -m src.predict applicants.csv --model nn
"""

import argparse
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data import GMSC_COLUMNS, RAW_FEATURES, ROOT
from src.models import predict_nn

MODELS_DIR = ROOT / "models"
NN_DIR = MODELS_DIR / "neural_network"


@lru_cache
def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "metadata.json").read_text())


@lru_cache
def load_model(kind: str = "best"):
    if kind == "nn":
        import keras

        pre = joblib.load(NN_DIR / "preprocessor.joblib")
        return pre, keras.models.load_model(NN_DIR / "credit_nn.keras", compile=False)
    return joblib.load(MODELS_DIR / "credit_model.joblib")


def default_probability(df: pd.DataFrame, kind: str = "best") -> np.ndarray:
    df = df.rename(columns=GMSC_COLUMNS)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    X = df[RAW_FEATURES]
    if kind == "nn":
        return predict_nn(*load_model("nn"), X)
    return load_model("best").predict_proba(X)[:, 1]


def risk_band(proba: float, threshold: float) -> str:
    if proba >= threshold:
        return "High"
    if proba >= threshold / 2:
        return "Medium"
    return "Low"


def score(df: pd.DataFrame, kind: str = "best") -> pd.DataFrame:
    meta = load_metadata()
    threshold = meta["nn_threshold"] if kind == "nn" else meta["threshold"]
    proba = default_probability(df, kind)
    return pd.DataFrame({
        "default_probability": proba.round(4),
        "predicted_default": (proba >= threshold).astype(int),
        "risk_band": [risk_band(p, threshold) for p in proba],
    }, index=df.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="input CSV with the raw applicant features")
    parser.add_argument("-o", "--output", help="where to write the scored CSV (default: print a preview)")
    parser.add_argument("--model", choices=["best", "nn"], default="best",
                        help="'best' = selected scikit-learn model, 'nn' = Keras neural network")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    scored = pd.concat([df, score(df, args.model)], axis=1)
    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"Scored {len(scored):,} rows -> {args.output}")
    else:
        print(scored.head(20).to_string())
    print("\nRisk bands:", scored["risk_band"].value_counts().to_dict())


if __name__ == "__main__":
    main()
