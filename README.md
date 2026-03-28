# 🌿 LeafScan — Potato Leaf Disease Detector

A production-deployed deep learning web app that detects potato leaf diseases from images using **MobileNetV2 + TFLite** — optimized for real-world deployment on constrained cloud infrastructure.

**Live Demo → [leaf-disease-detection-2-0.onrender.com](https://leaf-disease-detection-2-0.onrender.com)**



https://github.com/user-attachments/assets/74f0d74d-58f0-44ce-87a8-4eff00c901bc



---

## What it does

Upload or capture a photo of a potato leaf and LeafScan instantly classifies it into one of three categories:

| Class | Description |
|---|---|
| 🟡 Early Blight | *Alternaria solani* — dark spots with concentric rings |
| 🔴 Late Blight | *Phytophthora infestans* — water-soaked lesions, major crop threat |
| 🟢 Healthy | No visible disease markers |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV2 (pretrained ImageNet) + fine-tuned on PlantVillage dataset |
| Inference | TensorFlow Lite Float16 — 4.5 MB, ~120 MB RAM |
| Backend | Flask · Gunicorn |
| Frontend | Vanilla JS · CSS · Jinja2 |
| Deployment | Render (free tier) |
| Training | Google Colab (T4 GPU) |

---

## Architecture

```
User (browser)
    │
    ├── Upload / drag-drop / 📷 camera capture
    │
    ▼
Flask (app.py)
    │
    ├── /predict          → save upload → run_model()
    └── /predict-sample   → load from static/samples/ → run_model()
                                    │
                                    ▼
                        ai-edge-litert Interpreter
                        mobilenet_model.tflite (4.5 MB)
                        Input: (1, 224, 224, 3) · float32
                        Output: (1, 3) · softmax
                                    │
                                    ▼
                        JSON { label, confidence, image_url }
                                    │
                                    ▼
                        Frontend → animated pipeline → result card
```

---

## Model Details

### Training Strategy — Transfer Learning (2-phase)

**Phase 1 — Feature extraction (5 epochs)**
- MobileNetV2 base frozen
- Only classification head trained
- Adam lr=1e-3

**Phase 2 — Fine-tuning (10 epochs)**
- Top 30 layers of MobileNetV2 unfrozen
- Adam lr=1e-4 (10x lower)
- EarlyStopping + ReduceLROnPlateau

**Dataset** — PlantVillage (potato subset)
- 2,152 images · 3 classes
- 80/20 train/val split
- Augmentation: rotation, flip, zoom, shift

**Conversion** — TFLite Float16
- Keras model → `tf.lite.TFLiteConverter` → Float16 quantization
- Size: ~4.5 MB · Accuracy loss: <0.5%

---

## Why MobileNetV2 + TFLite over a custom CNN?

The original project used a custom 3-layer CNN with full TensorFlow:

| Metric | Custom CNN + TF | MobileNetV2 + TFLite |
|---|---|---|
| Model size | 38 MB | **4.5 MB** |
| RAM at inference | ~500 MB | **~120 MB** |
| Cold start | 30–60s → 502 errors | **~2–3s** |
| Val accuracy | 95.4% | **~97%** |
| Render free tier | ❌ OOM crashes | ✅ Works reliably |

Transfer learning from ImageNet weights gave both better accuracy and dramatically better deployment characteristics.

---

## Project Structure

```
leaf-disease-detection/
├── app.py                      # Flask app — TFLite inference
├── mobilenet_model.tflite      # Trained model (4.5 MB)
├── requirements.txt            # ai-edge-litert, flask, gunicorn, numpy, pillow
├── Procfile                    # gunicorn --timeout 120 --workers 1 --preload
├── templates/
│   └── index.html              # Full frontend — pipeline animation, camera, results
├── static/
│   └── samples/                # 9 sample leaf images (3 per class)
│       ├── early_blight/
│       ├── late_blight/
│       └── healthy/
└── cnn_model.ipynb             # Original CNN training notebook
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/Aadityayadav333/leaf-disease-detection-2.0.git
cd leaf-disease-detection-2.0

# Install (Python 3.10)
pip install -r requirements.txt

# Run
python app.py
# → http://localhost:10000
```

> **Note:** `ai-edge-litert` installs on Linux/macOS. On Windows, the app falls back to `tensorflow.lite.python.interpreter` automatically — full TensorFlow must be installed locally.

---

## Features

- **Drag & drop** image upload
- **📷 Camera capture** — desktop (getUserMedia) and mobile (native camera)
- **Sample images** — 9 pre-loaded leaf photos across all 3 classes
- **Animated pipeline** — live visualization of each inference step
- **Confidence bar** — softmax probability displayed for the predicted class
- **Parallel fetch** — network request fires simultaneously with animation (no waiting)

---

## Deployment Notes

Deployed on **Render free tier** (512 MB RAM, sleeps after 15 min inactivity).

Key decisions that make this work on free tier:
- `ai-edge-litert` instead of full TensorFlow — saves ~380 MB RAM
- `--preload` gunicorn flag — model loaded once at startup, not per-request
- `--workers 1` — prevents dual-model OOM on 512 MB instances
- UptimeRobot pinging every 10 min — prevents cold starts

---

## What I learned

- Transfer learning is not just an accuracy improvement — it's a deployment strategy. MobileNetV2's lightweight design was specifically engineered for resource-constrained environments.
- RAM profiling matters as much as model accuracy for real deployments. A 95% accurate model that OOMs in production is useless.
- TFLite Float16 quantization is effectively lossless for 3-class image classification — the confidence gap between classes is large enough that minor numerical differences never change the argmax.
- `Promise.all()` for parallel animation + API fetch prevents perceived latency from compounding with actual server latency.

---

## Author

**Aaditya Yadav** — B.Tech CSE  
[GitHub](https://github.com/Aadityayadav333) · [LinkedIn](https://linkedin.com/in/aaditya-yadav)

*"Code to know oneself and others better."*
