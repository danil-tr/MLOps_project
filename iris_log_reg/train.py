import os

import constants
import hydra
import joblib
import pandas as pd
from config import IrisData, Params
from dvc.api import DVCFileSystem
from hydra.core.config_store import ConfigStore
from sklearn.linear_model import LogisticRegression


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_iris", node=IrisData)


def train_model(train_df: pd.DataFrame, **model_params) -> LogisticRegression:
    train_df.drop(columns=train_df.columns[0], axis=1, inplace=True)
    X = train_df.iloc[:, 0:4]
    y = train_df["Species"]

    model = LogisticRegression(**model_params)
    model.fit(X, y)

    return model


@hydra.main(
    config_path=os.path.join(constants.get_project_path(), "config"),
    config_name="config",
    version_base="1.3.2",
)
def main(cfg: Params) -> None:
    project_directory_path = constants.get_project_path()

    data_path = os.path.join(project_directory_path, cfg["data"]["path"], "train.csv")
    model_path = os.path.join(project_directory_path, "model_result", "trained_model.sav")

    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/train.csv", data_path)
    train_df = pd.read_csv(data_path)

    model_params = {key: value for key, value in cfg["model"].items() if key != "name"}
    model = train_model(train_df, **model_params)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    main()
