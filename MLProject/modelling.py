import mlflow
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
)

def main():

    data = pd.read_csv('data_clean_sample.csv')


    # Feature engineering
    if "dest_freq" not in data.columns:
        if "nameDest" in data.columns:
            data["dest_freq"] = data["nameDest"].map(data["nameDest"].value_counts())
        else:
            raise KeyError("dest_freq or nameDest not found")

    if "orig_freq" not in data.columns:
        if "nameOrig" in data.columns:
            data["orig_freq"] = data["nameOrig"].map(data["nameOrig"].value_counts())
        else:
            raise KeyError("orig_freq or nameOrig not found")

    drop_cols = [c for c in ["nameDest", "nameOrig"] if c in data.columns]
    X = data.drop(columns=["isFraud"] + drop_cols)
    y = data["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Manual logging (AMAN UNTUK CI)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("max_depth", 8)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
