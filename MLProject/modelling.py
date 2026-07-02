import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model():
    # Gunakan dataset Churn, bukan Iris!
    df = pd.read_csv('churn_preprocessing.csv')
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Kriteria 2 Basic: Menggunakan autolog MLflow
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Basic_RandomForest_Churn"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Akurasi model Churn: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()
