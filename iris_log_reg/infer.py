import os

import constants
import hydra
import joblib
import pandas as pd
from config import IrisData, Params
from dvc.api import DVCFileSystem
from hydra.core.config_store import ConfigStore
from sklearn import metrics


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_iris", node=IrisData)


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


@hydra.main(
    config_path=os.path.join(constants.get_project_path(), "config"),
    config_name="config",
    version_base="1.3.2",
)
def main(cfg: Params) -> None:
    project_directory_path = constants.get_project_path()

    data_path = os.path.join(project_directory_path, cfg["data"]["path"], "test.csv")
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
