"""Lab 3.4: train four incident detectors, track them in MLflow, register the best.

Run from the lab folder:  python ml/train_models.py
Keep the ambiguous transition rows:  KEEP_TRANSITIONS=1 python ml/train_models.py
"""
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["req_rate", "err_rate", "p95_latency", "cpu"]
DATA = "data/incident_dataset.csv"
EXPERIMENT = "incident-detection"
MODEL_NAME = "ops-incident-detector"


class IncidentDetector(mlflow.pyfunc.PythonModel):
    """Uniform contract: 4 feature columns in, 0 (normal) / 1 (incident) out."""

    def __init__(self, model, is_anomaly_model=False):
        self.model = model
        self.is_anomaly_model = is_anomaly_model

    def predict(self, context, model_input, params=None):
        pred = self.model.predict(model_input[FEATURES])
        if self.is_anomaly_model:  # IsolationForest: -1 anomaly, +1 normal
            return (pred == -1).astype(int)
        return pred.astype(int)


def load_split():
    df = pd.read_csv(DATA).sort_values("ts")
    if os.environ.get("KEEP_TRANSITIONS") != "1" and "transition" in df.columns:
        df = df[~df["transition"].astype(bool)]
    df = df.reset_index(drop=True)
    split = int(len(df) * 0.5)  # time-based 50/50 split, never shuffled
    return df.iloc[:split], df.iloc[split:]


def scores(y_true, y_pred):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    train, test = load_split()
    keep = os.environ.get("KEEP_TRANSITIONS") == "1"

    candidates = []
    for c in (0.05, 0.2):
        m = IsolationForest(contamination=c, random_state=42)
        m.fit(train.loc[train["is_incident"] == 0, FEATURES])  # learn "normal" only
        candidates.append((f"isolation_forest_c{c}", m, True, {"contamination": c}))

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr.fit(train[FEATURES], train["is_incident"])
    candidates.append(("logistic_regression", lr, False, {"class_weight": "balanced"}))

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(train[FEATURES], train["is_incident"])
    candidates.append(("random_forest", rf, False, {"n_estimators": 100, "class_weight": "balanced"}))

    best = None
    for name, model, is_anomaly, params in candidates:
        detector = IncidentDetector(model, is_anomaly)
        result = scores(test["is_incident"], detector.predict(None, test))
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params, "keep_transitions": keep, "train_rows": len(train), "test_rows": len(test)})
            mlflow.log_metrics(result)
            info = mlflow.pyfunc.log_model(name="model", python_model=detector)
        print(name, result)
        if best is None or result["f1"] > best[1]["f1"]:
            best = (info.model_uri, result)

    version = mlflow.register_model(best[0], MODEL_NAME)
    print(f"Registered best model (f1={best[1]['f1']:.3f}) as {MODEL_NAME} v{version.version}")


if __name__ == "__main__":
    main()
