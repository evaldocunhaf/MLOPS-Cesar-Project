from abc import ABC

class BaseModel(ABC):

    def __init__(self, model):
        self.model = model


    def train(self):
        pass

    def calculate_metrics(self):
        pass