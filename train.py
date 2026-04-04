import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from trainer import SklearnTrainer

# ------------------------------------------------------------------
# Load params
# ------------------------------------------------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DATA_PATH       = params["data"]["processed_path"]
TEST_SIZE       = params["data"]["test_size"]
RANDOM_STATE    = params["data"]["random_state"]
TARGET_COL      = params["data"]["target_column"]
FEATURES        = params["data"]["features"]
MODEL_NAME      = params["train"]["model"]
EXPERIMENT_NAME = params["train"]["experiment_name"]
TRACKING_URI    = params["mlflow"]["tracking_uri"]
ARTIFACT_PATH   = params["mlflow"]["model_artifact_path"]
MODELS_DIR      = Path(params["output"]["models_dir"])
METRICS_DIR     = Path(params["output"]["metrics_dir"])

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
X = df[FEATURES]
y = df[TARGET_COL]

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ------------------------------------------------------------------
# Build estimator
# ------------------------------------------------------------------
def build_estimator(name: str, params: dict):
    if name == "random_forest":
        p = params["random_forest"]
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            class_weight=p["class_weight"],
            n_jobs=p["n_jobs"],
            random_state=params["data"]["random_state"],
        )
    elif name == "knn":
        p = params["knn"]
        return KNeighborsClassifier(
            n_neighbors=p["n_neighbors"],
            weights=p["weights"],
            metric=p["metric"],
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier
        p = params["xgboost"]
        return XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            eval_metric=p["eval_metric"],
            random_state=params["data"]["random_state"],
        )
    else:
        raise ValueError(f"Unknown model: {name}")


estimator = build_estimator(MODEL_NAME, params)

# XGBoost requires numeric labels
label_encoder = None
if MODEL_NAME == "xgboost":
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_fit = label_encoder.fit_transform(y_train)
    y_test_enc  = label_encoder.transform(y_test)
else:
    y_train_fit = y_train
    y_test_enc  = y_test

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
trainer = SklearnTrainer(estimator, categorical_cols, numeric_cols)
pipeline = trainer.train(X_train, y_train_fit)
y_pred_raw = trainer.predict(X_test)

if label_encoder is not None:
    y_pred = label_encoder.inverse_transform(y_pred_raw)
    y_test_eval = y_test
else:
    y_pred = y_pred_raw
    y_test_eval = y_test

metrics = trainer.calculate_metrics(y_test_eval, y_pred)
print("Metrics:", metrics)

# ------------------------------------------------------------------
# MLflow logging
# ------------------------------------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

model_params = params.get(MODEL_NAME, {})

# ------------------------------------------------------------------
# Serialize model (before MLflow so we can log the artifact)
# ------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / "model_pipeline.joblib"
joblib.dump(pipeline, model_path)
if label_encoder is not None:
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
print(f"Model saved to {model_path}")

with mlflow.start_run():
    mlflow.log_params({"model": MODEL_NAME, "test_size": TEST_SIZE, "random_state": RANDOM_STATE})
    mlflow.log_params(model_params)
    mlflow.log_metrics(metrics)
    try:
        mlflow.log_artifact(str(model_path), artifact_path=ARTIFACT_PATH)
    except Exception as e:
        print(f"Warning: could not upload artifact to MLflow server ({e}). Model is saved locally at {model_path}")

# ------------------------------------------------------------------
# DVC metrics
# ------------------------------------------------------------------
METRICS_DIR.mkdir(parents=True, exist_ok=True)
metrics_path = METRICS_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics written to {metrics_path}")
