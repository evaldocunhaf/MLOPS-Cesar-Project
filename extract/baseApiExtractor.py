import pandas as pd
from abc import ABC, abstractmethod

class BaseExtract(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def extract(self):
        pass