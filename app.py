from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("model/rice_leaf_model.h5")

# Classes and disease reasons with solutions
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).resize((128,128))  # match model input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    reason = disease_reasons[predicted_class]

    # Create list of tuples for table (class, confidence)
    classes_confidences = list(zip(class_names, (predictions*100).tolist()))

    return render_template(
        'result.html',
        prediction=predicted_class,
        reason=reason,
        image_path=url_for('static', filename='uploads/' + file.filename),
        classes_confidences=classes_confidences,
        classes=class_names,
        confidences=(predictions*100).tolist()
    )

if __name__ == '__main__':
    app.run(debug=True)
