from flask import Flask, request, redirect
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import os
import gdown

app = Flask(__name__)

# -----------------------------
# Configurations
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'rice_leaf_model.h5'
DRIVE_URL = 'https://drive.google.com/uc?id=1bXm5RthubHsGGelEIca3ZlAmh1UfE02v'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Download model if not found
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# -----------------------------
# Disease classes and reasons
# -----------------------------
class_names = [
    'Bacterial Blight', 'Brown Spot', 'Healthy',
    'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'
]

disease_reasons = {
    'Bacterial Blight': 'Caused by bacteria; leads to water-soaked lesions.<br>Solution: Use disease-resistant varieties, apply copper-based fungicides.',
    'Brown Spot': 'Caused by fungus; appears as brown spots.<br>Solution: Maintain proper fertilization and remove infected leaves.',
    'Healthy': 'No disease detected.<br>Solution: Maintain field hygiene and monitor regularly.',
    'Leaf Blast': 'Caused by fungus; diamond-shaped lesions.<br>Solution: Proper spacing, nitrogen management, and fungicide application.',
    'Leaf Scald': 'Caused by fungus; pale lesions along edges.<br>Solution: Crop rotation, balanced fertilization, and fungicides.',
    'Narrow Brown Spot': 'Caused by fungus; narrow brown lesions.<br>Solution: Maintain soil moisture, avoid excess nitrogen.'
}

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Smart Agriculture: Rice Leaf Disease Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            h1 {
                color: #2e7d32;
                font-size: 28px;
                text-align: center;
            }
            form {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                text-align: center;
                width: 320px;
            }
            input[type=file] {
                padding: 10px;
                margin: 15px 0;
                border-radius: 8px;
                border: 1px solid #ccc;
                width: 100%;
            }
            input[type=submit] {
                background-color: #43a047;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type=submit]:hover {
                background-color: #2e7d32;
            }
        </style>
    </head>
    <body>
        <h1>🌾 Smart Agriculture:<br>A Hybrid AI System for Early Detection of Rice Leaf Disease</h1>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <input type="submit" value="Upload & Predict">
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)

    img = Image.open(filepath).resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    reason = disease_reasons[predicted_class]

    del model

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #f1f8e9, #dcedc8);
                text-align: center;
                padding: 40px;
            }}
            h2 {{
                color: #1b5e20;
                font-size: 26px;
            }}
            h3 {{
                color: #2e7d32;
                font-size: 22px;
            }}
            p {{
                font-size: 18px;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
            }}
            img {{
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }}
            a {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background-color: #43a047;
                color: white;
                border-radius: 8px;
                text-decoration: none;
            }}
            a:hover {{
                background-color: #2e7d32;
            }}
        </style>
    </head>
    <body>
        <h2>Prediction Result</h2>
        <h3>{predicted_class}</h3>
        <p>{reason}</p>
        <img src="/{filepath}" width="300"><br>
        <a href="/">🔙 Go Back</a>
    </body>
    </html>
    """

# -----------------------------
# Render/Local Port Setup
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
