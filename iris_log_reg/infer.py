import os
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

def infer_model(model, test_df: pd.DataFrame) -> pd.DataFrame:
    test_df.drop(columns=test_df.columns[0], axis=1, inplace=True)  
    X = test_df.iloc[:, 0:4]
    y = test_df["Species"]

    prediction = model.predict(X)
    print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y))
    return prediction


if __name__ == "__main__":
    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    data_path = os.path.join(current_directory, "data", "Iris", "test.csv")
    model_path = os.path.join(current_directory, "data", "trained_model.sav")
    prediction_path = os.path.join(current_directory, "data", "prediction.csv")

    test_df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    prediction = pd.DataFrame(infer_model(model, test_df))
    
    pd.DataFrame(prediction).to_csv(prediction_path, sep=",", header=False, encoding="utf-8")
