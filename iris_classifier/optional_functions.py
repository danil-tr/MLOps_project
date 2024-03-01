import os
from dvc.api import DVCFileSystem
import pandas as pd


def get_project_path() -> str:
    current_file_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    return project_path


def get_dvc_data(fs: DVCFileSystem, dvc_file_path: str, file_path: str) -> pd.DataFrame:
    try:
        os.remove(file_path)
    except OSError:
        pass
    fs.get_file(dvc_file_path, file_path)
    return pd.read_csv(file_path)


if __name__ == "__main__":
    pass
