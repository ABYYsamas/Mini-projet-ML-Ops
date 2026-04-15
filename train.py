from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import json

def main():
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
 
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
 
    os.makedirs("artifacts", exist_ok=True)
 
    joblib.dump(model, os.path.join("artifacts", "model.pkl"))
    joblib.dump(scaler, os.path.join("artifacts", "scaler.pkl"))
 
    metrics = {"accuracy": float(acc), "f1_score": float(f1)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)
 
    print("Model saved to artifacts/model.pkl")
    print("Scaler saved to artifacts/scaler.pkl")
 
 
if __name__ == "__main__":
    main()