from glycan_prediction_model.dataset.query import query
from glycan_prediction_model.dataset.files import assign_file_paths
from glycan_prediction_model.dataset.density import create_maps
from glycan_prediction_model.dataset.split import test_train_split

from glycan_prediction_model.constants import DATASET_BASE_DIR
import shutil
import polars as pl


def cleanup_dataset_dir():
    removal_list = []
    for path in DATASET_BASE_DIR.glob("*"):
        if not path.is_dir():
            continue

        if not any(path.iterdir()):
            removal_list.append(path)

    for removal in removal_list:
        shutil.rmtree(str(removal))


# def create_dataset():
#     df = query()
#     df = assign_file_paths(df)
#     df = create_maps(df)
#     cleanup_dataset_dir()
#     train, test = test_train_split(df, split=0.8)
#     return train, test

def load_precomputed_dataset(): 
    print("Loading precomputed dataset")
    dataset = []

    for path in DATASET_BASE_DIR.glob("*"):
        source_path = path / "source.map"
        target_path = path / "target.map"
        help_path = path / "help.csv"

        if not source_path.exists() or not target_path.exists() or not help_path.exists():
            continue
        
        pdb = path.name
        dataset.append(
            {"pdb": pdb, 
             "source_path": str(source_path), 
             "target_path": str(target_path), 
             "help_path": str(help_path),
             })
        
    return pl.DataFrame(dataset)

def create_dataset(): 
    df = load_precomputed_dataset()
    train, test = test_train_split(df, split=0.8)
    return train, test

if __name__ == "__main__":
    create_dataset()
