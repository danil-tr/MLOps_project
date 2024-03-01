import os

import optional_functions
import hydra
import pandas as pd
from catboost import CatBoostClassifier
from config import IrisData, Params
from dvc.api import DVCFileSystem
from hydra.core.config_store import ConfigStore
from sklearn import metrics


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_iris", node=IrisData)


def infer_model(model: CatBoostClassifier, test_df: pd.DataFrame) -> pd.DataFrame:
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)
    X_test = test_df.iloc[:, 0:4]
    y_test = test_df["Species"]

    y_prediction = model.predict(X_test)

    print(
        "The accuracy of the model is",
        metrics.accuracy_score(y_prediction, y_test),
    )

    return y_prediction


@hydra.main(
    config_path=os.path.join(optional_functions.get_project_path(), "config"),
    config_name="config",
    version_base="1.3.2",
)
def main(cfg: Params) -> None:
    project_directory_path = optional_functions.get_project_path()
    data_path = os.path.join(project_directory_path, cfg["data"]["path"], "test.csv")
    model_path = os.path.join(
        project_directory_path, "model_result", "trained_model.onnx"
    )
    prediction_path = os.path.join(
        project_directory_path, "model_result", "prediction.csv"
    )

    fs = DVCFileSystem(project_directory_path)
    test_df = optional_functions.get_dvc_data(fs, "/data/test.csv", data_path)

    model = CatBoostClassifier()
    model.load_model(model_path, format="onnx")

    prediction = pd.DataFrame(infer_model(model, test_df))

    pd.DataFrame(prediction).to_csv(
        prediction_path, sep=",", header=False, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
