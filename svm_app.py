import joblib
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image

# Load the Random Forest model saved in joblib format/
# model = joblib.load(r"E:\flask_cnn_app\model\random_forest_folk_art_model.pkl")
model=joblib.load(r"E:\flask_cnn_app\model\svm_folk_art_model.pkl")

# Define class labels manually
class_labels = [
    'Aipan Art (Uttarakhand)', 'Assamese Miniature Painting (Assam)', 'Basholi Painting (Jammu and Kashmir)',
    'Bhil Painting (Madhya Pradesh)', 'Chamba Rumal (Himachal Pradesh)', 'Cheriyal Scroll Painting (Telangana)',
    'Dokra Art (West Bengal)', 'Gond Painting (Madhya Pradesh)', 'Kalamkari Painting (Andhra Pradesh and Telangana)',
    'Kalighat Painting (West Bengal)', 'Kangra Painting (Himachal Pradesh)', 'Kerala Mural Painting (Kerala)',
    'Kondapalli Bommallu (Andhra Pradesh)', 'Kutch Lippan Art (Gujarat)', 'Leather Puppet Art (Andhra Pradesh)',
    'Madhubani Painting (Bihar)', 'Mandala Art', 'Mandana Art (Rajasthan)', 'Mata Ni Pachedi (Gujarat)',
    'Meenakari Painting (Rajasthan)', 'Mughal Paintings', 'Mysore Ganjifa Art (Karnataka)',
    'Pattachitra Painting (Odisha and Bengal)', 'Patua Painting (West Bengal)', 'Pichwai Painting (Rajasthan)',
    'Rajasthani Miniature Painting (Rajasthan)', 'Rogan Art from Kutch (Gujarat)', 'Sohrai Art (Jharkhand)',
    'Tikuli Art (Bihar)', 'Warli Folk Painting (Maharashtra)'
]

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('svmindex.html')

# Image preprocessing function (resize to match model's expected input size)
def preprocess_image(image):
    # Resize the image to the expected input size (e.g., 8x16) based on model training dimensions
    image = image.resize((8, 16))  # Adjust to match the training input size
    image = image.convert('L')  # Convert to grayscale if needed
    image_array = np.array(image).flatten()  # Flatten the image to a 1D array of 128 elements
    image_array = image_array.reshape(1, -1)  # Reshape for model input
    return image_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the form
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
        
        # Open the image file
        image = Image.open(file.stream)
        
        # Preprocess the image
        input_array = preprocess_image(image)
        
        # Make prediction
        prediction_index = model.predict(input_array)[0]
        
        # Retrieve the class label
        prediction_label = class_labels[prediction_index]
        
        # Send result back to the page
        return render_template('svmindex.html', prediction_text=f'Predicted Class: {prediction_label}')
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
