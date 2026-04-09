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


def train_all_models():

    X, y = load_data()
    X_train, y_train, X_val, y_val = stratified_split(val_size=0.2)

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
        metrics = model.train(X_train, y_train, X_val, y_val)

        metrics["model"] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("f1_macro", ascending=False)

    df_results.to_csv("src/training/results/model_metrics_comparison")


train_all_models()
