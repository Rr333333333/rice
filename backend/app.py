from flask import Flask, request, redirect
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import load_model
import numpy as np
import os
import gdown
import cv2
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# -----------------------------
# Configurations
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'vgg19_rice_leaf_model.h5'
DRIVE_URL = 'https://drive.google.com/uc?id=1bXm5RthubHsGGelEIca3ZlAmh1UfE02v'  # replace if needed

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Download model if not found
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading VGG19 model...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# -----------------------------
# Class names and reasons
# -----------------------------
class_names = [
    'Bacterial Blight', 'Brown Spot', 'Healthy',
    'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'
]

disease_reasons = {
    'Bacterial Blight': 'Caused by bacteria; leads to water-soaked lesions.<br><b>Solution:</b> Use resistant varieties and copper-based sprays.',
    'Brown Spot': 'Caused by fungus; brown lesions on leaves.<br><b>Solution:</b> Use balanced fertilizer and remove infected leaves.',
    'Healthy': 'No disease detected.<br><b>Solution:</b> Maintain good field hygiene.',
    'Leaf Blast': 'Caused by fungus; diamond-shaped spots.<br><b>Solution:</b> Control nitrogen levels and apply fungicide.',
    'Leaf Scald': 'Caused by fungus; pale lesions along edges.<br><b>Solution:</b> Crop rotation and fungicide application.',
    'Narrow Brown Spot': 'Fungal disease with narrow brown streaks.<br><b>Solution:</b> Maintain soil moisture and avoid excessive nitrogen.'
}

# -----------------------------
# Grad-CAM Utility
# -----------------------------
def generate_gradcam(model, img_array, layer_name='block5_conv4'):
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(image_path, heatmap, output_path, alpha=0.5):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, superimposed_img)

# -----------------------------
# Home Page
# -----------------------------
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>🌾 Rice Leaf Disease Detection (VGG19 + XAI)</title>
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #f1f8e9, #dcedc8);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background: #fff;
                padding: 50px 60px;
                border-radius: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                text-align: center;
                width: 450px;
            }
            h1 {
                font-size: 22px;
                margin-bottom: 25px;
                color: #1b5e20;
            }
            label, input[type="submit"] {
                display: inline-block;
                padding: 15px 30px;
                background: #43a047;
                color: #fff;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                transition: 0.3s;
                margin-top: 10px;
            }
            label:hover, input[type="submit"]:hover { background: #2e7d32; }
            #file-name { margin-top: 15px; color: #333; font-size: 15px; }
            #preview { margin-top: 20px; max-width: 100%; border-radius: 12px; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌾 Rice Leaf Disease Classification (VGG19 + Grad-CAM)</h1>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <label for="file">Choose Leaf Image</label>
                <input type="file" id="file" name="file" accept="image/*" required hidden>
                <div id="file-name"></div>
                <img id="preview" alt="Preview">
                <input type="submit" value="Predict">
            </form>
        </div>
        <script>
            const fileInput = document.getElementById('file');
            const fileNameDisplay = document.getElementById('file-name');
            const preview = document.getElementById('preview');
            fileInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    fileNameDisplay.textContent = `Selected: ${file.name}`;
                    const reader = new FileReader();
                    reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
                    reader.readAsDataURL(file);
                }
            });
        </script>
    </body>
    </html>
    '''

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    import tensorflow as tf
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # -----------------------------
    # Image Preprocessing: Gaussian + CLAHE
    # -----------------------------
    img_cv = cv2.imread(filepath)
    img_cv = cv2.GaussianBlur(img_cv, (5, 5), 0)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Resize and normalize
    img = cv2.resize(img_cv, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # -----------------------------
    # Load Model and Predict
    # -----------------------------
    model = load_model(MODEL_PATH)
    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    max_prob = np.max(preds)

    if max_prob < 0.7:
        predicted_class = "Unknown / Not a Leaf"
        reason = "Image is unclear or not a rice leaf."
    else:
        predicted_class = class_names[pred_idx]
        reason = disease_reasons[predicted_class]

    # -----------------------------
    # Grad-CAM Visualization
    # -----------------------------
    heatmap = generate_gradcam(model, img_array, 'block5_conv4')
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f'gradcam_{file.filename}')
    overlay_gradcam(filepath, heatmap, gradcam_path)

    # -----------------------------
    # Return HTML
    # -----------------------------
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Poppins; background: #e8f5e9; text-align:center; }}
            h2 {{ color: #1b5e20; }}
            img {{ margin:20px; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.3); }}
            a {{ background:#43a047; color:white; padding:10px 20px; border-radius:8px; text-decoration:none; }}
        </style>
    </head>
    <body>
        <h2>Prediction Result</h2>
        <h3>{predicted_class}</h3>
        <p>{reason}</p>
        <img src="/{filepath}" width="300">
        <img src="/{gradcam_path}" width="300"><br>
        <a href="/">🔙 Go Back</a>
    </body>
    </html>
    """

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    import tensorflow as tf
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
