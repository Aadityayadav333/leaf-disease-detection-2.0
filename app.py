import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('plant_disease_model.h5')

# Must match your training order
CLASS_LABELS = ['Healthy', 'Early Blight', 'Late Blight']

DATASET_FOLDERS = {
    'late':    'Potato___Late_blight',
    'early':   'Potato___Early_blight',
    'healthy': 'Potato___healthy',
}
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def get_sample_images(folder_key, n=5):
    folder = DATASET_FOLDERS.get(folder_key, '')
    if not os.path.isdir(folder):
        return []
    files = [
        f for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1] in ALLOWED_EXT
    ][:n]
    return [{'filename': f, 'folder': folder} for f in files]


def run_model(file_path):
    img       = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds         = model.predict(img_array)
    predicted_idx = int(np.argmax(preds))
    confidence    = round(float(np.max(preds)) * 100, 2)
    return CLASS_LABELS[predicted_idx], confidence


# Serve dataset images directly (they are NOT in static/)
# GET /sample-image?folder=Potato___Late_blight&file=uuid___RS_LB.JPG
@app.route('/sample-image')
def sample_image():
    folder   = request.args.get('folder', '')
    filename = request.args.get('file', '')
    if folder not in DATASET_FOLDERS.values() or '..' in filename or '/' in filename:
        return 'Forbidden', 403
    return send_from_directory(folder, filename)


# Main page — sample filenames injected into Jinja2 template
@app.route('/', methods=['GET'])
def index():
    samples = {
        'late':    get_sample_images('late'),
        'early':   get_sample_images('early'),
        'healthy': get_sample_images('healthy'),
    }
    return render_template('index.html', samples=samples)


# Predict on user-uploaded image
# POST /predict  (multipart/form-data, field "file")
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    safe_name = os.path.basename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    f.save(file_path)

    label, confidence = run_model(file_path)

    return jsonify({
        'label':      label,
        'confidence': confidence,
        'image_url':  url_for('static', filename='uploads/' + safe_name),
        'filename':   safe_name,
    })


# Predict on a sample image from the dataset folders
# POST /predict-sample  (JSON: { folder, file })
@app.route('/predict-sample', methods=['POST'])
def predict_sample():
    data     = request.get_json()
    folder   = data.get('folder', '')
    filename = data.get('file', '')

    if folder not in DATASET_FOLDERS.values() or '..' in filename or '/' in filename:
        return jsonify({'error': 'Forbidden'}), 403

    file_path = os.path.join(folder, filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    label, confidence = run_model(file_path)

    return jsonify({
        'label':      label,
        'confidence': confidence,
        'filename':   filename,
        'folder':     folder,
        'view_url':   f'/sample-image?folder={folder}&file={filename}',
    })


if __name__ == '__main__':
    app.run(debug=True)
