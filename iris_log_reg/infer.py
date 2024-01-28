import os

import joblib
import pandas as pd
from dvc.api import DVCFileSystem
from sklearn import metrics


def infer_model(model, test_df: pd.DataFrame) -> pd.DataFrame:
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)
    X = test_df.iloc[:, 0:4]
    y = test_df["Species"]

    prediction = model.predict(X)
    print(
        "The accuracy of the Logistic Regression is",
        metrics.accuracy_score(prediction, y),
    )
    return prediction


def main():
    current_file_path = os.path.realpath(__file__)
    project_directory_path = os.path.dirname(os.path.dirname(current_file_path))

    data_path = os.path.join(project_directory_path, "data", "test.csv")
    model_path = os.path.join(project_directory_path, "model_result", "trained_model.sav")
    prediction_path = os.path.join(
        project_directory_path, "model_result", "prediction.csv"
    )

    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/test.csv", data_path)

    test_df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    prediction = pd.DataFrame(infer_model(model, test_df))

    pd.DataFrame(prediction).to_csv(
        prediction_path, sep=",", header=False, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
