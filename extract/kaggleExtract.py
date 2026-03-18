import shutil
from pathlib import Path

import kagglehub
from extract.baseApiExtractor import BaseExtract


class KaggleExtract(BaseExtract):
    def extract(self):
        dataset_dir = Path(kagglehub.dataset_download("shaistashahid/gaming-and-mental-health"))

        project_root = Path(__file__).resolve().parents[1]
        final_dir = project_root / "data/raw"
        final_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(dataset_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum .csv encontrado em: {dataset_dir}")

        copied = []
        for csv_path in csv_files:
            dest_path = final_dir / csv_path.name
            shutil.copy2(csv_path, dest_path)
            copied.append(dest_path)

        print("Dataset cache dir:", dataset_dir)
        print("Arquivos copiados para:", final_dir)
        for p in copied:
            print(" -", p)

        return copied