import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

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


class CatBoost(BaseModel):

    def __init__(
        self,
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        early_stopping_rounds=50,
        verbose=False,
        random_state=42,
    ):

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.model = None
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None):

        train_df = pd.DataFrame({"text": X_train})

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=self.random_state,
            verbose=self.verbose,
            allow_writing_files=False,
        )

        if X_val is not None and y_val is not None:

            val_df = pd.DataFrame({"text": X_val})

            self.model.fit(
                train_df,
                y_train,
                text_features=["text"],
                eval_set=(val_df, y_val),
            )

            self.is_trained = True
            metrics = self.evaluate(X_val, y_val)
            return metrics

        else:
            self.model.fit(
                train_df,
                y_train,
                text_features=["text"],
            )

        self.is_trained = True

        return None

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Сначала необходимо вызвать train()")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame({"text": X})
        elif isinstance(X, pd.Series):
            df = pd.DataFrame({"text": X.values})
        else:
            df = pd.DataFrame({"text": X})

        return self.model.predict(df)

    def predict_proba(self, X):

        if not self.is_trained:
            raise ValueError("Сначала необходимо вызвать train()")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame({"text": X})
        elif isinstance(X, pd.Series):
            df = pd.DataFrame({"text": X.values})
        else:
            df = pd.DataFrame({"text": X})

        return self.model.predict_proba(df)

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
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_trained = True

    def get_params(self):
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "loss_function": self.loss_function,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }


if __name__ == "__main__":
    from src.data.loader import load_data
    from src.data.splitter import stratified_split

    X, y = load_data()
    X_train, y_train, X_val, y_val = stratified_split(val_size=0.2)

    model = CatBoost(iterations=50)
    metrics = model.train(X_train, y_train, X_val, y_val)

    model.save("cat_boost.pkl")
