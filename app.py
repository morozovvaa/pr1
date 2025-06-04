from flask import Flask, request, jsonify
import joblib
import numpy as np
import gdown

app = Flask(__name__)

MODEL_PATH = "final_wine_model.pkl"
SCALER_PATH = "scaler.pkl"

# Загружаем модель из облачного хранилища, если её нет локально
file_id = "1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7"
url = f"https://drive.google.com/uc?id={file_id}"

output = "final_wine_model.pkl"
gdown.download(url, output, quiet=False)

# Загружаем объекты
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            data["fixed acidity"], data["volatile acidity"], data["citric acid"],
            data["residual sugar"], data["chlorides"], data["free sulfur dioxide"],
            data["total sulfur dioxide"], data["density"], data["pH"],
            data["sulphates"], data["alcohol"], data["type"]
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0].tolist()

    return jsonify({
        "prediction": prediction,
        "probabilities": probabilities
    })

@app.route("/")
def home():
    return "✅ Wine Quality Predictor API is running!"

if __name__ == "__main__":
    app.run(debug=True)
