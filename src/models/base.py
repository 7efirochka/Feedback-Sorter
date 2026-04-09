from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def evaluate(self, X, y):
        pass
