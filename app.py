from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
import pandas as pd

# Load the dataset
df = pd.read_csv('styles.csv')

# Define the mapping for the 7 categories
category_mapping = {
    'Shirts': 'Topwear', 'T-Shirts': 'Topwear', 'Blouses': 'Topwear',
    'Jeans': 'Bottomwear', 'Trousers': 'Bottomwear', 'Skirts': 'Bottomwear',
    'Sneakers': 'Footwear', 'Sandals': 'Footwear', 'Shoes': 'Footwear',
    'Belts': 'Accessories', 'Hats': 'Accessories', 'Scarves': 'Accessories',
    'Watches': 'Watches',
    'Earrings': 'Jewelry', 'Necklaces': 'Jewelry', 'Bracelets': 'Jewelry',
    'Lipstick': 'Beauty Products', 'Skincare': 'Beauty Products',
}

# Apply the mapping
df['masterCategory'] = df['articleType'].map(category_mapping).fillna('Other')

app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions (optional)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model = load_model('model.h5')

# Class names mapping (example)
class_names = {
    0: 'Apparel',
    1: 'Accessories',
    2: 'Footwear',
    3: 'Personal Care',
    4: 'Free Items',
    5: 'Sporting Goods',
    6: 'Home'
}

# Utility function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize image array if required by the model
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    
    # Debugging: Print the raw prediction output
    print(f"Raw prediction: {prediction}")
    
    # Get the predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names.get(predicted_class_index, "Unknown")
    
    return predicted_class_name


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            classification = predict_image(file_path)
            return render_template('result.html', classification=classification)
    return render_template('classify.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
