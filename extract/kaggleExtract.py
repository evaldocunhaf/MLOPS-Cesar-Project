import kagglehub
from baseApiExtractor import BaseExtract


class KaggleExtract(BaseExtract):
    def extract(self):
    # Download latest version
    path = kagglehub.dataset_download("shaistashahid/gaming-and-mental-health")

    print("Path to dataset files:", path)