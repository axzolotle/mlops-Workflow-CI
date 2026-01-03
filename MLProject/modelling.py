import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--random_state", type=int)
parser.add_argument("--n_estimators", type=int)
parser.add_argument("--max_depth", type=int)
args = parser.parse_args()

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Latihan Fraud Detection RandomForest")

data = pd.read_csv(args.data_path)

# (feature engineering kamu tetap)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state
)

mlflow.autolog()

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    random_state=args.random_state,
    max_depth=args.max_depth
)

model.fit(X_train, y_train)
