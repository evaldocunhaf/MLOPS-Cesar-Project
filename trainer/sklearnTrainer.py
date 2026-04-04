from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .baseTrainer import BaseTrainer


class SklearnTrainer(BaseTrainer):

    def __init__(self, estimator, categorical_cols: list, numeric_cols: list):
        self.estimator = estimator
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.pipeline = None

    def _build_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
                ("num", "passthrough", self.numeric_cols),
            ]
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.estimator),
        ])

    def train(self, X_train, y_train):
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def calculate_metrics(self, y_test, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
