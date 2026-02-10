"""
Brain Tumor Detection Web Application
Deployment-ready Flask app for Render
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow import keras
import base64
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'brain_tumor_classifier_final.h5'
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
    MODEL_LOADED = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image(image_path):
    """
    Make prediction on uploaded image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Prepare results
        result = {
            'class': class_names[predicted_class_idx],
            'confidence': f"{confidence * 100:.2f}",
            'all_probabilities': {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    
    # Check if model is loaded
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': result['class'],
            'confidence': result['confidence'],
            'all_probabilities': result['all_probabilities'],
            'image': img_data
        })
        
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html')


if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print("="*70)
    print("Brain Tumor Detection Web App")
    print(f"Model loaded: {MODEL_LOADED}")
    print(f"Running on port: {port}")
    print("="*70)
    
    # Run app
    app.run(host='0.0.0.0', port=port, debug=False)