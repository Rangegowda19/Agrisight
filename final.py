from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)

# Load models and class indices
models = {
    "predict_1": load_model('plant_model.keras'),
    "predict_2": load_model('plantdisease.keras')
}

class_indices = {}
for model_name in models:
    class_file = "class_indices.pkl" if model_name == "predict_1" else "class.pkl"
    with open(class_file, "rb") as f:
        class_indices[model_name] = pickle.load(f)

# Reverse the mapping
class_names = {model_name: {v: k for k, v in indices.items()} for model_name, indices in class_indices.items()}

@app.route('/')
def home():
    return "ML Model API is working! Available endpoints: /predict_1, /predict_2"

def predict_image(file, model_name):
    """Function to process and predict image"""
    if file.filename == '':
        return {"error": "No selected file"}, 400

    try:
        img = Image.open(file.stream)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the selected model
        model = models[model_name]
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[model_name][predicted_class_index]

        return {'model': model_name, 'predictedClass': predicted_class}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}, 500

@app.route('/predict_1', methods=['POST'])
def predict_1():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    return jsonify(predict_image(request.files['image'], "predict_1"))

@app.route('/predict_2', methods=['POST'])
def predict_2():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    return jsonify(predict_image(request.files['image'], "predict_2"))

if __name__ == '__main__':
    app.run(debug=True)
