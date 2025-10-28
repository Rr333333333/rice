from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import os
import gdown

app = Flask(__name__)

# Create folders for static uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Download model from Google Drive if missing
# -----------------------------
MODEL_PATH = "rice_leaf_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1bXm5RthubHsGGelEIca3ZlAmh1UfE02v"  # public link

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Classes and reasons
# -----------------------------
class_names = ['Bacterial Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

disease_reasons = {
    'Bacterial Blight': 'Caused by bacteria X; leads to water-soaked lesions.\nSolution: Use disease-resistant varieties, apply copper-based fungicides, avoid overhead irrigation.',
    'Brown Spot': 'Caused by fungus Y; appears as brown spots on leaves.\nSolution: Maintain proper fertilization, remove infected leaves, apply fungicide if necessary.',
    'Healthy': 'No disease detected. Leaf is healthy.\nSolution: Keep monitoring, maintain proper field hygiene.',
    'Leaf Blast': 'Caused by fungus Z; leads to diamond-shaped lesions.\nSolution: Proper spacing, nitrogen management, and fungicide application.',
    'Leaf Scald': 'Caused by fungus A; leads to pale lesions along leaf edges.\nSolution: Crop rotation, balanced fertilization, use fungicides if outbreak occurs.',
    'Narrow Brown Spot': 'Caused by fungus B; narrow brown lesions appear.\nSolution: Maintain soil moisture, avoid excessive nitrogen, and apply fungicides if needed.'
}

@app.route('/')
def index():
    return '''
    <h2>🌾 Rice Leaf Disease Detection</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload and Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = Image.open(filepath).resize((128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    reason = disease_reasons[predicted_class]

    response = f"""
    <h3>Prediction: {predicted_class}</h3>
    <p>{reason}</p>
    <img src="/{filepath}" width="250">
    <br><br>
    <a href="/">Go Back</a>
    """
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
