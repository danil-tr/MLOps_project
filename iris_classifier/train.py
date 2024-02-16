import os

import constants
import hydra
import mlflow
import pandas as pd
import project_metrics
from catboost import CatBoostClassifier
from config import IrisData, Params
from dvc.api import DVCFileSystem
from hydra.core.config_store import ConfigStore
from sklearn import metrics


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_iris", node=IrisData)


def train_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Params = None, **model_params
) -> CatBoostClassifier:
    X_train = train_df.iloc[:, 0:4]
    y_train = train_df["Species"]

    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train)

    mlflow.set_tracking_uri(cfg["tracking_server"]["uri"])
    mlflow.set_experiment(cfg["tracking_server"]["experiment_name"])
    with mlflow.start_run():
        mlflow.set_tags(tags={"version": 1.0, "framework": "Catboost"})

        features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        target = "Species"

        X_test = test_df.iloc[:, 0:4]
        y_test = test_df["Species"]
        y_prediction = model.predict(X_test)
        y_prediction_proba = model.predict_proba(X_test)

        roc_auc = metrics.roc_auc_score(
            y_test, y_prediction_proba, multi_class="ovr", average="macro"
        )
        precision = metrics.precision_score(y_test, y_prediction, average="weighted")
        recall = metrics.recall_score(y_test, y_prediction, average="weighted")
        f1 = metrics.f1_score(y_test, y_prediction, average="weighted")

        corr_matrix_fig = project_metrics.get_fig_correlation_matrix(
            pd.concat([X_test, X_train], ignore_index=True)
        )
        pair_features_fig = project_metrics.get_fig_pair_features(
            pd.concat([test_df, train_df], ignore_index=True)
        )

        # local_artifacts_path = os.path.join(constants.get_project_path(), "model_result")
        mlflow.log_params(
            {
                "features": features,
                "target": target,
                "model_type": model.__class__,
                "model_params": model.get_all_params(),
            }
        )
        mlflow.log_metrics(
            {"roc_auc": roc_auc, "precision": precision, "recall": recall, "f1": f1}
        )

        mlflow.log_figure(corr_matrix_fig, "correlation_matrix.png")
        mlflow.log_figure(pair_features_fig, "pair_features_fig.png")

    return model


@hydra.main(
    config_path=os.path.join(constants.get_project_path(), "config"),
    config_name="config",
    version_base="1.3.2",
)
def main(cfg: Params) -> None:
    project_directory_path = constants.get_project_path()
    data_train_path = os.path.join(
        project_directory_path, cfg["data"]["path"], "train.csv"
    )
    data_test_path = os.path.join(project_directory_path, cfg["data"]["path"], "test.csv")
    model_path = os.path.join(
        project_directory_path, "model_result", "trained_model.onnx"
    )

    fs = DVCFileSystem(project_directory_path)
    fs.get_file("/data/train.csv", data_train_path)
    fs.get_file("/data/test.csv", data_test_path)
    train_df, test_df = pd.read_csv(data_train_path), pd.read_csv(data_test_path)
    train_df.drop(columns=train_df.columns[0], axis=1, inplace=True)
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)

    model_params = {key: value for key, value in cfg["model"].items() if key != "name"}
    model = train_model(train_df, test_df, cfg, **model_params)
    model.save_model(model_path, format="onnx")


if __name__ == "__main__":
    main()
