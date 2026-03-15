import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for

# ── LiteRT (ai-edge-litert on Render) — full TF fallback locally ─
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load TFLite model once at startup ────────────────────────────
interpreter = Interpreter(model_path="mobilenet_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_H = input_details[0]['shape'][1]  # 224
INPUT_W = input_details[0]['shape'][2]  # 224

# Alphabetical folder order Keras used during training:
# Potato___Early_blight → 0
# Potato___Late_blight  → 1
# Potato___healthy      → 2
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]

SAMPLE_IMAGES = {
    "Early Blight": [
        "samples/early_blight/early1.jpg",
        "samples/early_blight/early2.jpg",
        "samples/early_blight/early3.jpg"
    ],
    "Late Blight": [
        "samples/late_blight/late1.jpg",
        "samples/late_blight/late2.jpg",
        "samples/late_blight/late3.jpg"
    ],
    "Healthy": [
        "samples/healthy/healthy1.jpg",
        "samples/healthy/healthy2.jpg",
        "samples/healthy/healthy3.jpg"
    ]
}


def run_model(file_path):
    img = Image.open(file_path).convert("RGB").resize((INPUT_W, INPUT_H))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_idx = int(np.argmax(preds))
    confidence    = round(float(np.max(preds)) * 100, 2)

    return CLASS_LABELS[predicted_idx], confidence


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_IMAGES)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext       = os.path.splitext(f.filename)[1].lower()
    safe_name = str(uuid.uuid4()) + ext
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    f.save(file_path)

    try:
        label, confidence = run_model(file_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "label":      label,
        "confidence": confidence,
        "image_url":  url_for("static", filename="uploads/" + safe_name),
        "filename":   f.filename or safe_name
    })


@app.route("/predict-sample", methods=["POST"])
def predict_sample():
    data     = request.get_json()
    img_path = data.get("img_path")
    if not img_path:
        return jsonify({"error": "Missing img_path"}), 400

    img_path  = img_path.replace("\\", "/").lstrip("/")
    file_path = os.path.join("static", img_path)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        label, confidence = run_model(file_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "label":      label,
        "confidence": confidence,
        "filename":   os.path.basename(img_path),
        "view_url":   url_for("static", filename=img_path)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
