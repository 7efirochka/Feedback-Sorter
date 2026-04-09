import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from .base import BaseModel
from pathlib import Path


class XGBoost(BaseModel):

    def __init__(
        self,
        max_features=10000,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_class=3,
        objective="multi:softprob",
        eval_metric="mlogloss",
    ):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.eval_metric = eval_metric
        self.num_class = num_class
        self.vectorizer = None
        self.model = None
        self.random_state = 42
        self.is_trained = False

    def create_pipeline(self):
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        norm="l2",
                        encoding="utf-8",
                    ),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        objective=self.objective,
                        num_class=self.num_class,
                        random_state=self.random_state,
                        eval_metric=self.eval_metric,
                    ),
                ),
            ]
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):

        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
            return metrics

        return None

    def predict(self, X):
        if self.is_trained:
            return self.pipeline.predict(X)

        raise ValueError("Сначала необходимо вызвать train()")

    def predict_proba(self, X):
        if self.is_trained:
            return self.pipeline.predict_proba(X)

        raise ValueError("Сначала необходимо вызвать train()")

    def evaluate(self, X, y):
        if not self.is_trained:
            raise ValueError("Сначала необходимо вызвать train()")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "precision_macro": precision_score(
                y, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }

        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def save(self, path):
        if not self.is_trained:
            raise ValueError("Нельзя сохранить необученную модель")
        joblib.dump(self.pipeline, path)

    def load(self, path):
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.is_trained = True

    def get_params(self):
        return {
            "max_features": self.max_features,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_state": self.random_state,
        }


if __name__ == "__main__":
    from src.data.loader import load_data
    from src.data.splitter import stratified_split

    X, y = load_data()
    X_train, y_train, X_val, y_val = stratified_split(val_size=0.2)

    model = XGBoost()
    metrics = model.train(X_train, y_train, X_val, y_val)

    model.save("xgboost_model.pkl")
