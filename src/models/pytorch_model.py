import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .base import BaseModel


class TextClassifier(nn.Module):
    """Нейросетевая модель для классификации тональности"""

    def __init__(self, input_dim, num_classes=3, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class PyTorchModel(BaseModel):

    def __init__(
        self,
        max_features=10000,
        hidden_dims=[128, 64],
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=30,
        patience=5,
        random_state=42,
        device=None,
    ):
        self.max_features = max_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.save = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vectorizer = None
        self.model = None
        self.is_trained = False

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _create_vectorizer(self):
        return TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            norm="l2",
            encoding="utf-8",
            sublinear_tf=True,
        )

    def _create_dataloaders(self, X_train, y_train, X_val=None, y_val=None):
        self.vectorizer = self._create_vectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(
            y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long
        )

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tfidf = self.vectorizer.transform(X_val)
            X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
            y_val_tensor = torch.tensor(
                y_val.values if hasattr(y_val, "values") else y_val, dtype=torch.long
            )

            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        return train_loader, val_loader, X_train_tensor.shape[1]

    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_loader, val_loader, input_dim = self._create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        y_unique = np.unique(y_train.values if hasattr(y_train, "values") else y_train)
        self._num_classes = len(y_unique)

        self.model = TextClassifier(
            input_dim=input_dim,
            num_classes=self._num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        self._fit(
            self.model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            self.device,
            epochs=self.epochs,
            patience=self.patience,
        )
        self.is_trained = True

        if X_val is not None and y_val is not None:
            return self.evaluate(X_val, y_val)
        return None

    def _train_one_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in loader:
            x = x_batch.to(device)
            y = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x = x_batch.to(device)
                y = y_batch.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def _fit(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=30,
        patience=5,
    ):
        best_val_acc = 0.0
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            if val_loader:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }

                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        model.to(device)

    def predict(self, X):

        if not self.is_trained or self.vectorizer is None:
            raise ValueError("Сначала необходимо вызвать train()")

        self.model.eval()

        X_tfidf = self.vectorizer.transform(X)
        X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

        return preds

    def predict_proba(self, X):

        if not self.is_trained or self.vectorizer is None:
            raise ValueError("Сначала необходимо вызвать train()")

        self.model.eval()

        X_tfidf = self.vectorizer.transform(X)
        X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

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
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        return metrics

    def save(self, path):
        if not self.is_trained:
            raise ValueError("Нельзя сохранить необученную модель")

        self.model.cpu()
        joblib.dump(self, path)
        self.model.to(self.device)

    @classmethod
    def load(cls, path):
        instance = joblib.load(path)
        instance.is_trained = True
        instance.model.to(instance.device)
        return instance

    def get_params(self):
        return {
            "max_features": self.max_features,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": str(self.device),
        }


if __name__ == "__main__":
    from src.data.loader import load_data
    from src.data.splitter import stratified_split

    X, y = load_data()
    X_train, y_train, X_val, y_val = stratified_split(val_size=0.2)

    model = PyTorchModel(
        max_features=5000,  # Для PyTorch лучше меньше признаков (скорость)
        hidden_dims=[128, 64],
        dropout=0.3,
        epochs=20,
        batch_size=64,
    )

    metrics = model.train(X_train, y_train, X_val, y_val)
    model.save("src/saved_models/PyTorch.pkl")
