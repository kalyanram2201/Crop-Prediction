import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
with open("crop_recommendation.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    crop_result = None
    if request.method == "POST":
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(features)
            crop_result = le.inverse_transform(prediction)[0]

        except Exception as e:
            crop_result = f"Error: {str(e)}"

    return render_template("index.html", crop=crop_result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
