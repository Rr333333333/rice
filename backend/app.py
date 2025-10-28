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
# Disease classes and details
# -----------------------------
class_names = [
    'Bacterial Blight', 'Brown Spot', 'Healthy',
    'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'
]

disease_reasons = {
    'Bacterial Blight': 'Caused by bacteria; leads to water-soaked lesions.<br><b>Solution:</b> Use disease-resistant varieties, apply copper-based fungicides.',
    'Brown Spot': 'Caused by fungus; appears as brown spots.<br><b>Solution:</b> Maintain proper fertilization and remove infected leaves.',
    'Healthy': 'No disease detected.<br><b>Solution:</b> Maintain field hygiene and monitor regularly.',
    'Leaf Blast': 'Caused by fungus; diamond-shaped lesions.<br><b>Solution:</b> Proper spacing, nitrogen management, and fungicide application.',
    'Leaf Scald': 'Caused by fungus; pale lesions along edges.<br><b>Solution:</b> Crop rotation, balanced fertilization, and fungicides.',
    'Narrow Brown Spot': 'Caused by fungus; narrow brown lesions.<br><b>Solution:</b> Maintain soil moisture, avoid excess nitrogen.'
}

# -----------------------------
# Home Page (Upload)
# -----------------------------
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Smart Agriculture: A Hybrid AI System for Early Detection of Rice Leaf Disease</title>
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
            input[type="file"] {
                display: none;
            }
            label {
                display: inline-block;
                padding: 15px 30px;
                background: #43a047;
                color: #fff;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                transition: 0.3s;
            }
            label:hover {
                background: #2e7d32;
            }
            input[type="submit"] {
                padding: 15px 35px;
                background: #1b5e20;
                color: #fff;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                transition: 0.3s;
                margin-top: 20px;
            }
            input[type="submit"]:hover {
                background: #2e7d32;
            }
            p.note {
                margin-top: 20px;
                color: #555;
                font-size: 14px;
            }
            #file-name {
                margin-top: 15px;
                color: #333;
                font-size: 15px;
            }
            #preview {
                margin-top: 20px;
                max-width: 100%;
                border-radius: 12px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌾 Smart Agriculture: A Hybrid AI System for Early Detection of Rice Leaf Disease</h1>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <label for="file">Choose Leaf Image</label>
                <input type="file" id="file" name="file" accept="image/*" required>
                <div id="file-name"></div>
                <img id="preview" alt="Image Preview">
                <br>
                <input type="submit" value="Predict">
            </form>
            <p class="note">Supported formats: JPEG, PNG</p>
        </div>

        <script>
            const fileInput = document.getElementById('file');
            const fileNameDisplay = document.getElementById('file-name');
            const preview = document.getElementById('preview');

            fileInput.addEventListener('change', function() {
                const file = this.files[0];
                if (file) {
                    fileNameDisplay.textContent = `Selected file: ${file.name}`;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                } else {
                    fileNameDisplay.textContent = '';
                    preview.style.display = 'none';
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
    max_prob = np.max(predictions)
    predicted_index = np.argmax(predictions)

    if max_prob < 0.7:
        predicted_class = "Not a Leaf / Unknown"
        reason = "The uploaded image doesn't resemble a rice leaf. Please upload a clear image of a rice leaf."
    else:
        predicted_class = class_names[predicted_index]
        reason = disease_reasons[predicted_class]

    del model

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #f1f8e9, #dcedc8);
                text-align: center;
                padding: 40px;
            }}
            h2 {{ color: #1b5e20; font-size: 26px; }}
            h3 {{ color: #2e7d32; font-size: 22px; }}
            p {{ font-size: 18px; color: #333; max-width: 600px; margin: 0 auto; }}
            img {{ margin-top: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }}
            a {{
                display: inline-block;
                margin-top: 30px;
                padding: 10px 20px;
                background-color: #43a047;
                color: white;
                border-radius: 8px;
                text-decoration: none;
            }}
            a:hover {{ background-color: #2e7d32; }}
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
# Render-specific Port Setup
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
