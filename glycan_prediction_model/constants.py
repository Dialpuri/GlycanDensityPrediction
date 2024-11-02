import enum
from pathlib import Path

PDB_QUERY_URL = "https://www.ebi.ac.uk/pdbe/search/pdb/select?"
PDB_BASE_DIR = Path("/vault/pdb_mirror/data/structures/all/")

BASE_DIR = Path("glycan_prediction_model")
CACHE_DIR = BASE_DIR / ".cache"

DATASET_BASE_DIR = Path("dataset")

DATASET_PATH = Path("dataset.csv")

class DataSources(enum.Enum):
    local = 1
    api = 2


DATA_SOURCE = DataSources.local
