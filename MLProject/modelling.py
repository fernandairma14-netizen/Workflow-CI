import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_model():
    # Configure MLflow
    mlflow.set_experiment("Diabetes_Prediction_Irma")
    
    # Load preprocessed data
    if not os.path.exists("diabetes_preprocessing.csv"):
        print("Preprocessed data not found.")
        return
        
    df = pd.read_csv("diabetes_preprocessing.csv")
    
    # Split features and target (Outcome)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # KRITERIA 2 BASIC: Menggunakan autolog sepenuhnya
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Basic_RandomForest_Diabetes"):
        # Train baseline model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Baseline Model trained successfully! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
