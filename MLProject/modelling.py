import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Latihan Fraud Detection RandomForest")

# Load preprocessed dataset
data = pd.read_csv("/Users/axzolotle/Code/Self-project/mlops-subs/membangun_model/data_clean.csv")

# Feature engineering 
# Compute frequency features only if they are not already present.
if "dest_freq" not in data.columns:
    if "nameDest" in data.columns:
        data["dest_freq"] = data["nameDest"].map(data["nameDest"].value_counts())
    else:
        raise KeyError(
            "Neither 'dest_freq' nor 'nameDest' found in data. Re-run preprocessing or provide the raw columns."
        )

if "orig_freq" not in data.columns:
    if "nameOrig" in data.columns:
        data["orig_freq"] = data["nameOrig"].map(data["nameOrig"].value_counts())
    else:
        raise KeyError(
            "Neither 'orig_freq' nor 'nameOrig' found in data. Re-run preprocessing or provide the raw columns."
        )

# Prepare features and target. Drop identifier columns only if present.
drop_cols = [c for c in ["nameDest", "nameOrig"] if c in data.columns]
X = data.drop(columns=["isFraud"] + drop_cols)
y = data["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

input_example = X_train.iloc[:5]

# Training + MLflow Tracking
mlflow.autolog()

n_estimators = 100
random_state = 42
max_depth = 23

model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
    max_depth=max_depth
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


