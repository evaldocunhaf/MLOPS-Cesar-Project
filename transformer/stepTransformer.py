import pandas as pd

from .baseTransformer import BaseTransformer
from pathlib import Path

class StepTransformer(BaseTransformer):
    def __init__(self, columns: list, target_column: str):
        self.columns = columns
        self.target_column = target_column

    def transform(self, data_path: str):
        df = pd.read_csv(data_path)
        df1 = df[self.columns].copy()

        mapping = {
            "Failing": "Low",
            "Poor": "Low",
            "Below Average": "Medium",
            "Average": "Medium",
            "Good": "High",
            "Excellent": "High",
        }

        df1[self.target_column] = df1[self.target_column].map(mapping)

        project_root = Path(__file__).resolve().parents[1]
        final_dir = project_root / "data/processed"
        df1.to_parquet(final_dir / 'data.parquet')
        df1.to_csv(final_dir / 'data.csv', index=False)
