"""SageMaker training script (Lab 4.2 locally, Lab 4.3 as a training job).

The SageMaker contract:
  - input data is read from the `train` channel  -> SM_CHANNEL_TRAIN
  - the model is written to                      -> SM_MODEL_DIR
  - metrics are printed as `f1=0.123` so SageMaker can scrape them with a regex
  - model_fn() tells the hosting container how to load the model
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

FEATURES = ["req_rate", "err_rate", "p95_latency", "cpu"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "incident_dataset.csv")).sort_values("ts")
    if "transition" in df.columns:
        df = df[~df["transition"].astype(bool)]
    df = df.reset_index(drop=True)
    split = int(len(df) * 0.5)
    train, test = df.iloc[:split], df.iloc[split:]

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, class_weight="balanced")
    model.fit(train[FEATURES], train["is_incident"])
    pred = model.predict(test[FEATURES])

    print(f"precision={precision_score(test['is_incident'], pred, zero_division=0):.3f}")
    print(f"recall={recall_score(test['is_incident'], pred, zero_division=0):.3f}")
    print(f"f1={f1_score(test['is_incident'], pred, zero_division=0):.3f}")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Called by the SageMaker sklearn serving container to load the model."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def predict_fn(input_data, model):
    """A single CSV row arrives as a 1-D array; the model expects rows x 4 features."""
    return model.predict(np.atleast_2d(np.asarray(input_data, dtype=float)))


if __name__ == "__main__":
    main()
