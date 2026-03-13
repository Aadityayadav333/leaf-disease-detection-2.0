import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("plant_disease_model.keras", compile=False)

# Sample images served from static/samples/
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

# Must match training order
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]


# ========================
# MODEL PREDICTION FUNCTION
# ========================
def run_model(file_path):
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)

    predicted_idx = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)

    return CLASS_LABELS[predicted_idx], confidence


# ========================
# HOME PAGE
# ========================
@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_IMAGES)


# ========================
# USER IMAGE PREDICTION
# ========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    import uuid
    ext = os.path.splitext(f.filename)[1].lower()
    safe_name = str(uuid.uuid4()) + ext

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    f.save(file_path)

    try:
        label, confidence = run_model(file_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "label": label,
        "confidence": confidence,
        "image_url": url_for("static", filename="uploads/" + safe_name),
        "filename": f.filename or safe_name
    })


# ========================
# SAMPLE IMAGE PREDICTION
# Sample images live in static/samples/<subfolder>/<file>
# ========================
@app.route("/predict-sample", methods=["POST"])
def predict_sample():
    data = request.get_json()

    # imgPath from JS is like "samples/early_blight/early1.jpg"
    img_path = data.get("img_path")

    if not img_path:
        return jsonify({"error": "Missing img_path"}), 400

    # Sanitize — prevent path traversal
    img_path = img_path.replace("\\", "/").lstrip("/")
    file_path = os.path.join("static", img_path)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        label, confidence = run_model(file_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    filename = os.path.basename(img_path)
    view_url = url_for("static", filename=img_path)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "filename": filename,
        "view_url": view_url
    })


# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
