from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def calculate_metrics(self, y_test, y_pred) -> dict:
        pass
