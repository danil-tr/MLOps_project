import os

import constants
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


images_path = os.path.join(constants.get_project_path(), "model_result")


def get_fig_correlation_matrix(iris_df: pd.DataFrame) -> plt.figure:
    plt.figure(figsize=(8, 4))
    sns.heatmap(iris_df.corr(), annot=True, fmt=".0%")
    return plt.gcf()


def get_fig_pair_features(iris_df: pd.DataFrame) -> plt.figure:
    plt.figure(figsize=(16, 16))
    sns.pairplot(iris_df.iloc[:, :], hue="Species")
    return plt.gcf()
