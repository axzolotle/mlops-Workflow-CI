import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Latihan_Fraud_Detection_RandomForest")

    # Aktifkan autolog sebelum proses training dimulai
    mlflow.autolog()

    data = pd.read_csv(args.data_path)
        
    df_numeric = data.select_dtypes(include=['number'])
        
    if 'is_fraud' in df_numeric.columns:
        target_col = 'is_fraud'
    elif 'Class' in df_numeric.columns:
        target_col = 'Class'
    else:
            # Jika nama kolom beda, ambil kolom terakhir sebagai target
        target_col = df_numeric.columns[-1]

    X = df_numeric.drop(target_col, axis=1)
    y = df_numeric[target_col]
    # 5. Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

        # 6. Initialize & Train Model
        # Parameter diambil dari argumen MLproject
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1 # Gunakan semua core CPU agar lebih cepat
    )

    model.fit(X_train, y_train)

        # 7. Tambahan: Log Metrik Manual (Optional karena sudah ada autolog)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log f1-score untuk kelas fraud (asumsi label 1 adalah fraud)
    if "1" in report:
        mlflow.log_metric("f1_score_fraud", report["1"]["f1-score"])

if __name__ == "__main__":
    main()
