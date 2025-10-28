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
    <h2>🌾 Rice Leaf Disease Detection</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <input type="submit" value="Upload & Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Validate file
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Lazy load model here to reduce memory usage
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)

    # Preprocess image
    img = Image.open(filepath).resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    reason = disease_reasons[predicted_class]

    # Free up memory
    del model

    # Return response
    return f"""
    <h3>Prediction: {predicted_class}</h3>
    <p>{reason}</p>
    <img src="/{filepath}" width="250">
    <br><br>
    <a href="/">🔙 Go Back</a>
    """

# -----------------------------
# Render-specific port setup
# -----------------------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
