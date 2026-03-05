import pandas as pd
import numpy as np

from datetime import date

from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


class BaseTransformer(ABC):

    @abstractmethod
    def transform(self, df):
        pass
