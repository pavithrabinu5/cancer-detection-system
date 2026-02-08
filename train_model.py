"""
FRESH START - Brain Tumor Detection Training
Complete script to train from scratch and achieve 90%+ accuracy
Save this as: train_model.py
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

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("BRAIN TUMOR DETECTION - FRESH TRAINING")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\nSTEP 1: Loading data...")
print("-"*70)

data_dir = Path('brain_tumor_dataset/Training')
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

images = []
labels = []

for class_idx, class_name in enumerate(class_names):
    class_path = data_dir / class_name
    image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
    print(f"Loading {len(image_files)} images from {class_name}...")
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            images.append(img)
            labels.append(class_idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

X = np.array(images, dtype=np.float32)
y = np.array(labels)

print(f"\nTotal images loaded: {len(X)}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ============================================================================
# STEP 2: BUILD MODEL
# ============================================================================

print("\nSTEP 2: Building model...")
print("-"*70)

# Load pre-trained EfficientNetB0
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(224, 224, 3))
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
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built successfully!")
print(f"Total parameters: {model.count_params():,}")

# ============================================================================
# STEP 3: TRAIN MODEL (PHASE 1)
# ============================================================================

print("\nSTEP 3: Training model (Phase 1 - Transfer Learning)...")
print("-"*70)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    zoom_range=0.25,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train Phase 1
history1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 1 training complete!")

# ============================================================================
# STEP 4: FINE-TUNE MODEL (PHASE 2)
# ============================================================================

print("\nSTEP 4: Fine-tuning model (Phase 2 - Unfreezing layers)...")
print("-"*70)

# Unfreeze base model
base_model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Phase 2
history2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 2 fine-tuning complete!")

# ============================================================================
# STEP 5: EVALUATE MODEL
# ============================================================================

print("\nSTEP 5: Evaluating model...")
print("-"*70)

# Predict on test set
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate metrics
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================================
# STEP 6: SAVE MODEL AND VISUALIZATIONS
# ============================================================================

print("\nSTEP 6: Saving model and visualizations...")
print("-"*70)

# Save model
model.save('brain_tumor_classifier_final.h5')
print("✓ Model saved: brain_tumor_classifier_final.h5")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Combine both training phases
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

# Accuracy plot
axes[0].plot(all_acc, label='Train')
axes[0].plot(all_val_acc, label='Validation')
axes[0].axvline(x=len(history1.history['accuracy']), color='red', linestyle='--', label='Fine-tuning starts')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(all_loss, label='Train')
axes[1].plot(all_val_loss, label='Validation')
axes[1].axvline(x=len(history1.history['loss']), color='red', linestyle='--', label='Fine-tuning starts')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved: training_history.png")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved: confusion_matrix.png")

# ============================================================================
# DONE!
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print("\nFiles created:")
print("  1. brain_tumor_classifier_final.h5 - Trained model")
print("  2. training_history.png - Training curves")
print("  3. confusion_matrix.png - Performance matrix")
print("\nYou can now run: python app.py")
print("="*70)