"""
Web App for Brain Tumor Detection
Save as: app.py
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from brain_tumor_classifier import BrainTumorClassifier
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

classifier = BrainTumorClassifier()
try:
    classifier.load_model('brain_tumor_classifier_final.h5')
    print("✓ Model loaded!")
except:
    print("✗ Model not found. Train it first!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = classifier.predict_single_image(filepath)
        
        with open(filepath, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': result['class'],
            'confidence': f"{result['confidence'] * 100:.2f}",
            'all_probabilities': result['all_probabilities'],
            'image': img_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*70)
    print("Brain Tumor Detection Web App")
    print("Go to: http://localhost:5000")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)