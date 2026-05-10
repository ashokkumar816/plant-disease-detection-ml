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

# ── ML-style disease knowledge base ─────────────────────────────────────────
DISEASE_KB = {
    "Tomato": [
        {"name": "Early Blight",         "type": "Fungal",      "base_conf": 82, "keywords": ["brown","dark","spots","concentric","yellowing"]},
        {"name": "Late Blight",          "type": "Fungal",      "base_conf": 85, "keywords": ["water","soaked","dark","lesion","white","mold"]},
        {"name": "Leaf Curl Virus",      "type": "Viral",       "base_conf": 78, "keywords": ["curl","yellow","vein","mosaic","stunted"]},
        {"name": "Bacterial Wilt",       "type": "Bacterial",   "base_conf": 80, "keywords": ["wilt","droop","brown","stem","water"]},
        {"name": "Septoria Leaf Spot",   "type": "Fungal",      "base_conf": 76, "keywords": ["small","circular","spot","grey","brown","margin"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal","fine"]},
    ],
    "Potato": [
        {"name": "Late Blight",          "type": "Fungal",      "base_conf": 88, "keywords": ["dark","lesion","water","soaked","rot"]},
        {"name": "Early Blight",         "type": "Fungal",      "base_conf": 80, "keywords": ["brown","spots","concentric","yellowing"]},
        {"name": "Black Scurf",          "type": "Fungal",      "base_conf": 74, "keywords": ["black","scurf","tuber","surface","rough"]},
        {"name": "Common Scab",          "type": "Bacterial",   "base_conf": 72, "keywords": ["scab","rough","corky","lesion"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Pepper": [
        {"name": "Anthracnose",          "type": "Fungal",      "base_conf": 81, "keywords": ["sunken","dark","lesion","fruit","rot"]},
        {"name": "Bacterial Leaf Spot",  "type": "Bacterial",   "base_conf": 79, "keywords": ["water","soaked","yellow","halo","angular"]},
        {"name": "Phytophthora Blight",  "type": "Fungal",      "base_conf": 83, "keywords": ["wilt","collapse","dark","stem","rot"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Rice": [
        {"name": "Rice Blast",           "type": "Fungal",      "base_conf": 87, "keywords": ["diamond","lesion","grey","center","brown","border"]},
        {"name": "Brown Spot",           "type": "Fungal",      "base_conf": 80, "keywords": ["brown","circular","spot","yellow","halo"]},
        {"name": "Bacterial Leaf Blight","type": "Bacterial",   "base_conf": 82, "keywords": ["yellow","stripe","wilt","margin","water"]},
        {"name": "Sheath Blight",        "type": "Fungal",      "base_conf": 78, "keywords": ["sheath","white","lesion","oval","grey"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Maize": [
        {"name": "Northern Corn Leaf Blight","type":"Fungal",   "base_conf": 84, "keywords": ["long","grey","tan","lesion","cigar"]},
        {"name": "Common Rust",          "type": "Fungal",      "base_conf": 86, "keywords": ["rust","orange","pustule","powder","brown"]},
        {"name": "Grey Leaf Spot",       "type": "Fungal",      "base_conf": 79, "keywords": ["grey","rectangular","spot","tan","stripe"]},
        {"name": "Maize Streak Virus",   "type": "Viral",       "base_conf": 77, "keywords": ["streak","yellow","stripe","mosaic","stunted"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Cotton": [
        {"name": "Cotton Leaf Curl",     "type": "Viral",       "base_conf": 85, "keywords": ["curl","yellow","vein","thicken","stunted"]},
        {"name": "Alternaria Leaf Spot", "type": "Fungal",      "base_conf": 78, "keywords": ["brown","circular","spot","concentric"]},
        {"name": "Bacterial Blight",     "type": "Bacterial",   "base_conf": 80, "keywords": ["angular","water","soaked","brown","black"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Wheat": [
        {"name": "Wheat Rust",           "type": "Fungal",      "base_conf": 87, "keywords": ["rust","orange","yellow","pustule","stripe"]},
        {"name": "Powdery Mildew",       "type": "Fungal",      "base_conf": 83, "keywords": ["white","powder","coating","mildew"]},
        {"name": "Septoria Blotch",      "type": "Fungal",      "base_conf": 78, "keywords": ["brown","blotch","tan","lesion","irregular"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
    "Banana": [
        {"name": "Panama Wilt",          "type": "Fungal",      "base_conf": 88, "keywords": ["wilt","yellow","brown","vascular","rot"]},
        {"name": "Sigatoka Leaf Spot",   "type": "Fungal",      "base_conf": 82, "keywords": ["yellow","streak","brown","necrotic","border"]},
        {"name": "Bunchy Top Virus",     "type": "Viral",       "base_conf": 80, "keywords": ["stunted","bunchy","mosaic","yellow","narrow"]},
        {"name": "Healthy",              "type": "None",        "base_conf": 95, "keywords": ["healthy","green","normal"]},
    ],
}

# Confidence adjustments based on environmental factors
WEATHER_BOOST = {
    "Fungal":    {"humid": +8, "dry": -6, "moderate":  0, "cold": +3},
    "Bacterial": {"humid": +5, "dry": -3, "moderate":  0, "cold": -2},
    "Viral":     {"humid": +2, "dry": +2, "moderate":  0, "cold": -1},
    "None":      {"humid":  0, "dry":  0, "moderate":  0, "cold":  0},
}

STAGE_BOOST = {
    "Fungal":    {"seedling": -3, "vegetative": +2, "flowering": +5, "fruiting": +4},
    "Bacterial": {"seedling": -2, "vegetative": +1, "flowering": +3, "fruiting": +3},
    "Viral":     {"seedling": +4, "vegetative": +2, "flowering":  0, "fruiting": -1},
    "None":      {"seedling":  0, "vegetative":  0, "flowering":  0, "fruiting":  0},
}

SPREAD_BOOST = {
    "isolated":  -5,
    "spreading": +3,
    "whole":     +7,
    "stem":      +9,
    "field":     +10,
}


def ml_score_diseases(crop, symptoms, weather, stage, spread):
    """
    Rule-based ML scoring to pre-rank candidate diseases before Claude.
    Combines keyword matching + environmental factor boosts.
    Returns top 3 candidates with their adjusted confidence scores.
    """
    diseases  = DISEASE_KB.get(crop, DISEASE_KB["Tomato"])
    sym_lower = (symptoms or "").lower()
    scored    = []

    for d in diseases:
        score = d["base_conf"]
        dtype = d["type"]

        # Keyword match: each matching keyword adds 3 points, capped at +12
        matched = sum(1 for kw in d["keywords"] if kw in sym_lower)
        score  += min(matched * 3, 12)

        # Environmental boosts
        score += WEATHER_BOOST.get(dtype, {}).get(weather, 0)
        score += STAGE_BOOST.get(dtype, {}).get(stage, 0)
        score += SPREAD_BOOST.get(spread, 0)

        # Clamp to valid range
        score = max(40, min(99, score))
        scored.append({"name": d["name"], "type": dtype, "score": round(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:3]


def build_prompt(crop, symptoms, weather, stage, area, spread, lang, candidates):
    lang_note = "Respond in Tamil language." if lang == "ta" else "Respond in English."
    cand_text = "\n".join(
        f"  - {c['name']} ({c['type']}, pre-score: {c['score']}%)"
        for c in candidates
    )
    return f"""You are an expert plant pathologist AI with 20+ years of field experience.

INPUT DATA:
  Crop           : {crop}
  Symptoms       : {symptoms or 'None provided'}
  Weather        : {weather}
  Growth Stage   : {stage}
  Affected Area  : {area}%
  Spread Pattern : {spread}

ML PRE-ANALYSIS (top ranked candidates):
{cand_text}

Using all the input data and ML pre-analysis above, give your final expert diagnosis.
{lang_note}

Respond ONLY with a valid JSON object — no markdown, no extra text, no explanation outside the JSON:
{{
  "disease": "exact disease name",
  "confidence": <integer 60-99>,
  "severity": "Mild|Moderate|Severe",
  "severity_pct": <integer 10-90>,
  "disease_type": "Fungal|Bacterial|Viral|Nutritional|Pest|None",
  "treatment": "step-by-step treatment (2-3 sentences)",
  "prevention": "practical prevention tips (2-3 sentences)",
  "organic": "organic or natural remedy options (1-2 sentences)"
}}"""


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ── Collect all form fields ───────────────────────────────────────────────
    crop     = request.form.get("crop",     "Tomato")
    symptoms = request.form.get("symptoms", "")
    weather  = request.form.get("weather",  "moderate")
    stage    = request.form.get("stage",    "vegetative")
    area     = request.form.get("area",     "25")
    spread   = request.form.get("spread",   "isolated")
    lang     = request.form.get("lang",     "en")

    # ── Step 1: ML pre-scoring ────────────────────────────────────────────────
    candidates = ml_score_diseases(crop, symptoms, weather, stage, spread)

    # ── Step 2: Build message content ─────────────────────────────────────────
    content = []

    image_file = request.files.get("image")
    if image_file and image_file.filename:
        img_bytes = image_file.read()
        img_b64   = base64.standard_b64encode(img_bytes).decode("utf-8")
        mime_type = image_file.content_type or "image/jpeg"

        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("uploads", f"{timestamp}_{image_file.filename}")
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
        "text": build_prompt(crop, symptoms, weather, stage, area, spread, lang, candidates),
    })

    # ── Step 3: Call Claude API ───────────────────────────────────────────────
    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1024,
            messages   = [{"role": "user", "content": content}],
        )

        raw   = "".join(b.text for b in response.content if hasattr(b, "text"))
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)

    except json.JSONDecodeError:
        # Fallback: use top ML candidate directly
        top  = candidates[0]
        data = {
            "disease":      top["name"],
            "confidence":   top["score"],
            "severity":     "Moderate",
            "severity_pct": 50,
            "disease_type": top["type"],
            "treatment":    (
                "Apply appropriate fungicide or bactericide based on disease type. "
                "Remove and destroy all infected plant material. "
                "Ensure proper drainage and avoid waterlogging."
            ),
            "prevention":   (
                "Use certified disease-free seeds and resistant varieties. "
                "Maintain proper plant spacing for good air circulation. "
                "Practice crop rotation every season."
            ),
            "organic":      (
                "Spray neem oil solution (5ml per litre of water) every 7 days. "
                "Apply Trichoderma-based bio-fungicide at the root zone."
            ),
        }
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # ── Step 4: Update history ────────────────────────────────────────────────
    emoji_map = {
        "Fungal": "🍄", "Bacterial": "🦠", "Viral": "🧬",
        "Nutritional": "🌿", "Pest": "🐛", "None": "✅"
    }
    history.insert(0, {
        "time":       datetime.now().strftime("%d-%m-%Y %H:%M"),
        "disease":    data.get("disease", "Unknown"),
        "crop":       crop,
        "confidence": data.get("confidence", 0),
        "emoji":      emoji_map.get(data.get("disease_type", ""), "🔬"),
    })
    history[:] = history[:10]

    # ── Step 5: Return full response ──────────────────────────────────────────
    return jsonify({
        "disease":      data.get("disease"),
        "confidence":   data.get("confidence"),
        "severity":     data.get("severity"),
        "severity_pct": data.get("severity_pct"),
        "disease_type": data.get("disease_type"),
        "treatment":    data.get("treatment"),
        "prevention":   data.get("prevention"),
        "organic":      data.get("organic"),
        "candidates":   candidates,
        "history":      history,
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Fetch current scan history."""
    return jsonify(history)


@app.route("/history", methods=["DELETE"])
def clear_history_route():
    """Clear all scan history."""
    history.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
