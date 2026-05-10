from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import json
import re
from datetime import datetime
import anthropic

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

history = []

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(crop, symptoms, weather, stage, area, spread, lang):
    lang_note = "Respond in Tamil language." if lang == "ta" else "Respond in English."
    return f"""You are an expert plant pathologist AI. Analyze this plant and give a detailed disease diagnosis.

Crop: {crop}
Symptoms described: {symptoms or 'None provided'}
Weather: {weather}
Growth stage: {stage}
Affected area: {area}%
Spread: {spread}
{lang_note}

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
  "disease": "disease name",
  "confidence": <number 60-99>,
  "severity": "Mild|Moderate|Severe",
  "severity_pct": <number 10-90>,
  "disease_type": "Fungal|Bacterial|Viral|Nutritional|Pest",
  "treatment": "detailed treatment steps",
  "prevention": "prevention strategy",
  "organic": "organic/natural treatment options"
}}"""


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Collect form fields ──────────────────────────────────────────────────
    crop    = request.form.get("crop",     "Tomato")
    symptoms= request.form.get("symptoms", "")
    weather = request.form.get("weather",  "moderate")
    stage   = request.form.get("stage",    "vegetative")
    area    = request.form.get("area",     "25")
    spread  = request.form.get("spread",   "isolated")
    lang    = request.form.get("lang",     "en")

    # ── Build message content (image optional) ───────────────────────────────
    content = []

    image_file = request.files.get("image")
    if image_file and image_file.filename:
        img_bytes  = image_file.read()
        img_b64    = base64.standard_b64encode(img_bytes).decode("utf-8")
        mime_type  = image_file.content_type or "image/jpeg"

        # Save a copy locally for audit / logging
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path  = os.path.join("uploads", f"{timestamp}_{image_file.filename}")
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": mime_type,
                "data":       img_b64,
            },
        })

    content.append({
        "type": "text",
        "text": build_prompt(crop, symptoms, weather, stage, area, spread, lang),
    })

    # ── Call Claude ──────────────────────────────────────────────────────────
    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1000,
            messages   = [{"role": "user", "content": content}],
        )

        raw   = "".join(block.text for block in response.content if hasattr(block, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)

    except json.JSONDecodeError:
        # Fallback if model returns non-JSON
        data = {
            "disease":      f"{crop} Disease (Analysis Unavailable)",
            "confidence":   70,
            "severity":     "Moderate",
            "severity_pct": 50,
            "disease_type": "Unknown",
            "treatment":    "Consult a local agricultural officer for diagnosis.",
            "prevention":   "Maintain good field hygiene and balanced nutrition.",
            "organic":      "Neem oil spray can help manage mild infections.",
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ── Update history (keep last 10) ────────────────────────────────────────
    history.insert(0, {
        "time":       datetime.now().strftime("%d-%m-%Y %H:%M"),
        "disease":    data.get("disease", "Unknown"),
        "crop":       crop,
        "confidence": data.get("confidence", 0),
        "emoji":      {
            "Fungal":      "🍄",
            "Bacterial":   "🦠",
            "Viral":       "🧬",
            "Nutritional": "🌿",
            "Pest":        "🐛",
        }.get(data.get("disease_type", ""), "🔬"),
    })
    history[:] = history[:10]

    # ── Return full result ───────────────────────────────────────────────────
    return jsonify({
        "disease":      data.get("disease"),
        "confidence":   data.get("confidence"),
        "severity":     data.get("severity"),
        "severity_pct": data.get("severity_pct"),
        "disease_type": data.get("disease_type"),
        "treatment":    data.get("treatment"),
        "prevention":   data.get("prevention"),
        "organic":      data.get("organic"),
        "history":      history,
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Optional standalone endpoint to fetch scan history."""
    return jsonify(history)


@app.route("/history", methods=["DELETE"])
def clear_history():
    """Clear all history entries."""
    history.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
