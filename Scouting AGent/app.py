from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Pepper__Bacterial_spot",
    "Pepper__healthy",
    "Potato__Early_blight",
    "Potato__healthy",
    "Potato__Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_YellowLeaf_Curl_Virus"
]

history = []

def severity(conf):
    if conf >= 85: return "High"
    elif conf >= 70: return "Medium"
    return "Low"

@app.route("/")
def home():
    return send_file(INDEX_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image uploaded"})

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = image.load_img(path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)

    disease = CLASS_NAMES[idx]
    confidence = round(float(preds[idx] * 100), 2)

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
        "prevention": "Use healthy seeds and crop rotation",
        "history": history
    })

if __name__ == "__main__":
    app.run(debug=True)
# Data generators