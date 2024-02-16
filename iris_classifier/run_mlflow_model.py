import os

import constants
import hydra
import pandas as pd
import requests
from config import IrisData, Params
from dvc.api import DVCFileSystem
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_iris", node=IrisData)


@hydra.main(
    config_path=os.path.join(constants.get_project_path(), "config"),
    config_name="config",
    version_base="1.3.2",
)
def main(cfg: Params) -> None:
    project_directory_path = constants.get_project_path()

    data_path = os.path.join(project_directory_path, cfg["data"]["path"], "test.csv")
    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/test.csv", data_path)

    test_df = pd.read_csv(data_path)
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)
    X_test = test_df.iloc[:, 0:4]

    uri = f"{cfg['inference_server']['uri']}/invocations"

    json_data = {"dataframe_split": X_test.iloc[:5].to_dict(orient="split")}
    print(json_data)

    response = requests.post(uri, json=json_data)
    print(f"\nPredictions:\n{response.json()}")


if __name__ == "__main__":
    main()
