from extract import KaggleExtract
import os
os.environ["KAGGLEHUB_CACHE_DIR"] = "/Users/evaldocunhafilho/Documents/Data_Science/MLOPS/MLOPS-Cesar-Project/data"

extract = KaggleExtract()
extract.extract()