import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- KONFIGURASI ---
# GANTI DENGAN USERNAME DAN NAMA REPO DAGSHUB ANDA
DAGSHUB_USER = "Username_Anda"
DAGSHUB_REPO = "Nama_Repo_Anda"
DATA_PATH = "namadataset_preprocessing/clean_fraud_data.csv" # Sesuaikan nama file

def train_model():
    # 1. Setup DagsHub & MLflow
    # Ini otomatis mengonfigurasi MLflow tracking URI ke DagsHub
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("Fraud_Detection_Experiment")

    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Asumsi kolom target bernama 'class' atau 'fraud'. Sesuaikan!
    X = df.drop(columns=['class']) # Ganti 'class' dengan nama kolom target Anda
    y = df['class']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai MLflow Run
    with mlflow.start_run(run_name="RandomForest_Manual_Log"):
        
        # --- A. Define Hyperparameters ---
        n_estimators = 100
        max_depth = 10
        
        # --- B. Log Parameters (Manual) ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "Random Forest")

        # --- C. Train Model ---
        print("Training model...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)

        # --- D. Calculate Metrics ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted') # Pakai weighted jika multiclass/imbalance
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {acc}")

        # --- E. Log Metrics (Manual) ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- F. Log ARTIFACTS (Syarat: Minimal 2 Tambahan) ---
        
        # Artefak 1: Confusion Matrix Plot (Gambar)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan dulu ke file lokal sementara
        plot_filename = "confusion_matrix.png"
        plt.savefig(plot_filename)
        # Upload ke MLflow
        mlflow.log_artifact(plot_filename)
        print("Artifact 1 (Confusion Matrix) logged.")

        # Artefak 2: Classification Report (Text File)
        report = classification_report(y_test, y_pred)
        report_filename = "classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        # Upload ke MLflow
        mlflow.log_artifact(report_filename)
        print("Artifact 2 (Report Text) logged.")

        # --- G. Log Model ---
        mlflow.sklearn.log_model(model, "model")
        print("Model saved to DagsHub/MLflow.")

        # Bersihkan file temporary
        os.remove(plot_filename)
        os.remove(report_filename)

if __name__ == "__main__":
    train_model()