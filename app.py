"""
API Flask - Wine Prediction
- charge le modèle et le scaler
- expose un endpoint de santé GET /health
- expose un endpoint de prédiction POST /predict
"""

from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# Chemins vers les artefacts
MODEL_PATH = Path("artifacts/model.pkl")
SCALER_PATH = Path("artifacts/scaler.pkl")

# Si le modèle n'existe pas, on lance l'entraînement
if not MODEL_PATH.exists():
    import train as _train
    _train.main()

# Charger le modèle et le scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Vérifier que les données sont bien envoyées
    if not data or "features" not in data:
        return jsonify({"error": "Envoie un JSON avec la clé 'features'"}), 400

    features = data["features"]

    # Vérifier qu'on a bien 13 features
    if len(features) != 13:
        return jsonify({"error": "Il faut 13 features (caractéristiques du vin)"}), 400

    try:
        # Normaliser les données avec le scaler
        features_scaled = scaler.transform([features])

        # Prédire avec le modèle
        prediction = model.predict(features_scaled)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)