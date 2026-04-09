import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC as SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .base import BaseModel
from pathlib import Path


class LinearSVC(BaseModel):

    def __init__(
        self,
        max_features=10000,
        max_iter=10000,
        class_weight="balanced",
        loss="squared_hinge",
        C=1.0,
        random_state=42,
    ):

        self.max_features = max_features
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.loss = loss
        self.random_state = random_state
        self.C = C
        self.pipeline = None
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
                    CalibratedClassifierCV(
                        estimator=SVC(
                            C=self.C,
                            max_iter=self.max_iter,
                            class_weight=self.class_weight,
                            loss=self.loss,
                        ),
                        cv=5,
                        method="sigmoid",
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
        self.pipeline = joblib.load(path)
        self.is_trained = True

    def get_params(self):
        return {
            "max_features": self.max_features,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "loss": self.loss,
            "random_state": self.random_state,
        }


if __name__ == "__main__":
    from src.data.loader import load_data
    from src.data.splitter import stratified_split

    X, y = load_data()
    X_train, y_train, X_val, y_val = stratified_split(val_size=0.2)

    model = LinearSVC(max_features=2000)
    metrics = model.train(X_train, y_train, X_val, y_val)

    model.save("LinearSVC_model.pkl")
