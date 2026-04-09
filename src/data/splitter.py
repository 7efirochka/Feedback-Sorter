import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.loader import load_data


def stratified_split(val_size=0.2, random_state=42, stratify=True):
    X, y = load_data()

    # тестовые данные не нужны, так как модель будет применяться к данным, введенным пользователем.
    # валидация нужна, чтобы сделать csv файлы с результатами всех моделей и сравнительным анализом лучших.

    # перед использованием обучу модели на всех данных, для более точных предсказаний от пользователя

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )

    return X_train, y_train, X_val, y_val


def split_info():
    X_train, y_train, X_val, y_val = stratified_split()

    info = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "total_size": len(X_train) + len(X_val),
        "train_distribution": y_train.value_counts().sort_index().to_dict(),
        "val_distribution": y_val.value_counts().sort_index().to_dict(),
    }

    return info


def get_full_data():
    X, y = load_data()

    return X, y
