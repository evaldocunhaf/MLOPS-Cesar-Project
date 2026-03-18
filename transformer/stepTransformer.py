import pandas as pd

from .baseTransformer import BaseTransformer
from pathlib import Path

class StepTransformer(BaseTransformer):
    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, data_path: str):
        df = pd.read_csv(data_path)
        df1 = df[self.columns]
        project_root = Path(__file__).resolve().parents[1]
        final_dir = project_root / "data/raw"
        df1.to_parquet(final_dir / 'data.parquet')