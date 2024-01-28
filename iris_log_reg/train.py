import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(train_df: pd.DataFrame) -> LogisticRegression:
    train_df.drop(columns=train_df.columns[0], axis=1, inplace=True)
    X = train_df.iloc[:, 0:4]
    y = train_df["Species"]

    model = LogisticRegression(solver="lbfgs", max_iter=200)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    current_file_path = os.path.realpath(__file__)
    parent_directory_path = os.path.dirname(os.path.dirname(current_file_path))

    data_path = os.path.join(parent_directory_path, "data", "train.csv")
    model_path = os.path.join(parent_directory_path, "model_result", "trained_model.sav")

    train_df = pd.read_csv(data_path)
    model = train_model(train_df)
    joblib.dump(model, model_path)
