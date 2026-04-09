from pathlib import Path
import pandas as pd

from src.data.loader import load_data
from src.data.splitter import stratified_split

from src.models.logistic_regression_model import LogisticRegression
from src.models.linearSVC_model import LinearSVC
from src.models.xgboost_model import XGBoost
from src.models.catboost_model import CatBoost
from src.models.multinomialNB_model import MultinomialNB
from src.models.pytorch_model import PyTorchModel

from src.data.splitter import get_full_data


def train_all_models():

    X_full, y_full = get_full_data()

    models = {
        "LogisticRegression": LogisticRegression(max_features=10000),
        "LinearSVC": LinearSVC(max_features=10000),
        "MultinomialNB": MultinomialNB(max_features=10000),
        "XGBoost": XGBoost(max_features=10000, n_estimators=100),
        "CatBoost": CatBoost(iterations=100, verbose=False),
        "PyTorch": PyTorchModel(
            max_features=5000, epochs=10
        ),  # меньше эпох для скорости
    }

    results = []

    for name, model in models.items():
        metrics = model.train(X_full, y_full)

        save_path = f"src/saved_models/{name}.pkl"
        model.save(save_path)

    return


# print(train_all_models())
if __name__ == "__main__":
    train_all_models()
