import pandas as pd
from abc import ABC, abstractmethod

class BaseExtract(ABC):

    @abstractmethod
    def extract(self):
        pass