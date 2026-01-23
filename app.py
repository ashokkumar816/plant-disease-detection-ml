from flask import Flask, request, jsonify, send_from_directory
import os
from datetime import datetime

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

history = []

def severity(conf):
    if conf >= 85:
        return "High"
    elif conf >= 70:
        return "Medium"
    return "Low"

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop", "Tomato")
    disease = f"{crop} Early Blight"
    confidence = 88

    history.insert(0, {
        "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "disease": disease,
        "confidence": f"{confidence}%"
    })
    history[:] = history[:5]

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "severity": severity(confidence),
        "treatment": "Use recommended fungicide",
        "prevention": "Avoid overhead irrigation and remove infected leaves",
        "history": history
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
