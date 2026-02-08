"""
Brain Tumor Classification System
Save as: brain_tumor_classifier.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorClassifier:
    
    def __init__(self, img_size=(224, 224), batch_size=32, epochs=50):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
    def load_and_preprocess_data(self, data_dir):
        print("Loading data...")
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = Path(data_dir) / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} not found!")
                continue
                
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error: {e}")
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        print(f"\nTotal images: {len(X)}")
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self):
        print("\nBuilding model...")
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*self.img_size, 3))
        base_model.trainable = False
        
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        print("Model ready!")
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2)
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTRAINING COMPLETE!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        print("\nEvaluating...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return {'predictions': y_pred, 'probabilities': y_pred_proba}
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        print("Saved: training_history.png")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("Saved: confusion_matrix.png")
    
    def save_model(self, filepath='brain_tumor_classifier_final.h5'):
        self.model.save(filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath='brain_tumor_classifier_final.h5'):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")
    
    def predict_single_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {name: float(prob) for name, prob in zip(self.class_names, predictions[0])}
        }


def main():
    print("="*70)
    print("BRAIN TUMOR DETECTION SYSTEM")
    print("="*70)
    
    classifier = BrainTumorClassifier(img_size=(224, 224), batch_size=32, epochs=50)
    
    # UPDATE THIS PATH!
    data_dir = 'brain_tumor_dataset/Training'
    
    if not Path(data_dir).exists():
        print(f"\nERROR: Cannot find {data_dir}")
        print("Download dataset first!")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.load_and_preprocess_data(data_dir)
    model = classifier.build_model()
    model.summary()
    
    classifier.train(X_train, y_train, X_val, y_val)
    results = classifier.evaluate(X_test, y_test)
    
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_test, results['predictions'])
    classifier.save_model('brain_tumor_classifier_final.h5')
    
    print("\n" + "="*70)
    print("DONE! Run app.py to test it.")
    print("="*70)


if __name__ == "__main__":
    main()