# Triton doesn't support the output of the onnx catboost model
# Create model that
import os
from pathlib import Path

import pandas as pd
from dvc.api import DVCFileSystem
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier


def get_project_path() -> str:
    return Path(__file__).resolve().parents[4]


def train_and_save_model(train_df: pd.DataFrame) -> None:
    X_train = train_df.iloc[:, 0:4]
    y_train = train_df["Species"]

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    initial_types = [("float_input", FloatTensorType([None, 4]))]
    options = {id(model): {"zipmap": False, "output_class_labels": True}}
    onx = to_onnx(model, X_train, options=options, initial_types=initial_types)

    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())


def main() -> None:
    project_directory_path = get_project_path()
    data_train_path = os.path.join(project_directory_path, "data", "train.csv")

    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/train.csv", data_train_path)

    train_df = pd.read_csv(data_train_path)
    train_df.drop(columns=train_df.columns[0], axis=1, inplace=True)

    train_and_save_model(train_df)


if __name__ == "__main__":
    main()
