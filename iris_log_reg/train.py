import os

import joblib
import pandas as pd
from dvc.api import DVCFileSystem
from sklearn.linear_model import LogisticRegression


def train_model(train_df: pd.DataFrame) -> LogisticRegression:
    train_df.drop(columns=train_df.columns[0], axis=1, inplace=True)
    X = train_df.iloc[:, 0:4]
    y = train_df["Species"]

    model = LogisticRegression(solver="lbfgs", max_iter=200)
    model.fit(X, y)

    return model


def main():
    current_file_path = os.path.realpath(__file__)
    project_directory_path = os.path.dirname(os.path.dirname(current_file_path))

    data_path = os.path.join(project_directory_path, "data", "train.csv")
    model_path = os.path.join(project_directory_path, "model_result", "trained_model.sav")

    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/train.csv", data_path)
    train_df = pd.read_csv(data_path)
    model = train_model(train_df)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    main()
