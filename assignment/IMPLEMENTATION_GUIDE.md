# Step-by-Step Implementation Guide for LLM Agents

## Overview

This guide provides detailed instructions for implementing the 3rd Mandatory Assignment for the Neural Networks & Deep Learning course. 

**Implementation Choice**: **Convolutional Autoencoder** for CIFAR-10 image reconstruction and classification

**Framework**: **TensorFlow/Keras** (industry-standard, production-ready framework)

**Rationale**:
- ✅ **Industry Standard**: Autoencoders are widely used in production (anomaly detection, compression, feature learning)
- ✅ **Modern Architecture**: Demonstrates state-of-the-art deep learning techniques
- ✅ **Excellent Support**: TensorFlow/Keras provides robust tools, documentation, and built-in CIFAR-10 dataset
- ✅ **Project Alignment**: CIFAR-10 dataset built into Keras (`keras.datasets.cifar10`)
- ✅ **Comprehensive Evaluation**: Enables both reconstruction and classification tasks
- ✅ **Production Ready**: TensorFlow is the most widely deployed deep learning framework

---

## Phase 1: Project Setup & Data Preparation

### Step 1.1: Verify Environment
```bash
# Check Python version (requires 3.8+)
python3 --version

# Install required packages
pip install tensorflow numpy matplotlib scikit-learn pandas tqdm seaborn
```

### Step 1.2: Understand Project Structure
```
Project 3/
├── assignment/           # Assignment documentation
├── Cifar10/             # CIFAR-10 data directory (auto-downloaded by Keras)
├── data/                # Additional data storage
├── src/                 # Source code
├── results/             # Output results
├── Report/              # LaTeX report
└── legacy/              # Previous implementations
```

### Step 1.3: Load CIFAR-10 Dataset
Create file: `src/load_data.py`

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_cifar10():
    """
    Φόρτωση και προεπεξεργασία CIFAR-10
    (Load and preprocess CIFAR-10)
    """
    # Load CIFAR-10 dataset (built into Keras)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [-1, 1] range for tanh activation
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f'Training samples: {x_train.shape[0]}')
    print(f'Test samples: {x_test.shape[0]}')
    print(f'Image shape: {x_train.shape[1:]}')
    print(f'Classes: {len(np.unique(y_train))}')
    
    return (x_train, y_train), (x_test, y_test)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
```

---

## Phase 2: Implement Core Architecture

### Step 2.1: Create Autoencoder Model
Create file: `src/autoencoder.py`

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def build_autoencoder(latent_dim=128, input_shape=(32, 32, 3)):
    """
    Δημιουργία Convolutional Autoencoder για CIFAR-10
    (Create Convolutional Autoencoder for CIFAR-10)
    
    Args:
        latent_dim: Διάσταση του latent space (Latent space dimension)
        input_shape: Διαστάσεις εισόδου (Input dimensions)
    
    Returns:
        autoencoder, encoder, decoder models
    """
    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
    # Convolutional blocks
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', name='enc_conv1')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='enc_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', name='enc_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Flatten and create latent representation
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_space')(x)
    
    encoder = models.Model(encoder_input, latent, name='encoder')
    
    # ==================== DECODER ====================
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    
    # Dense layer to reshape
    x = layers.Dense(128 * 4 * 4, activation='relu')(decoder_input)
    x = layers.Reshape((4, 4, 128))(x)
    
    # Transposed convolutional blocks
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', name='dec_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name='dec_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='dec_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output layer with tanh activation (matches [-1, 1] normalization)
    decoder_output = layers.Conv2D(3, (3, 3), padding='same', activation='tanh', name='decoder_output')(x)
    
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    
    # ==================== COMPLETE AUTOENCODER ====================
    autoencoder_input = layers.Input(shape=input_shape, name='autoencoder_input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')
    
    return autoencoder, encoder, decoder
```

### Step 2.2: Implement Training Loop
Create file: `src/train_autoencoder.py`

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def train_autoencoder(autoencoder, x_train, x_test, epochs=50, batch_size=128, lr=1e-3):
    """
    Εκπαίδευση του Autoencoder
    (Train the Autoencoder)
    
    Args:
        autoencoder: Το μοντέλο autoencoder (The autoencoder model)
        x_train: Δεδομένα εκπαίδευσης (Training data)
        x_test: Δεδομένα ελέγχου (Test data)
        epochs: Αριθμός epochs (Number of epochs)
        batch_size: Μέγεθος batch (Batch size)
        lr: Ρυθμός μάθησης (Learning rate)
    
    Returns:
        history, training_time
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Compile model
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'results/best_autoencoder.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger('results/training_log.csv')
    ]
    
    # Track training time
    start_time = time.time()
    
    # Train
    print('\n' + '='*50)
    print('Έναρξη εκπαίδευσης (Starting training)')
    print('='*50 + '\n')
    
    history = autoencoder.fit(
        x_train, x_train,  # Input and target are the same for autoencoder
        validation_data=(x_test, x_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print('\n' + '='*50)
    print(f'Χρόνος εκπαίδευσης: {training_time:.2f} δευτερόλεπτα')
    print(f'Training time: {training_time:.2f} seconds')
    print('='*50 + '\n')
    
    return history, training_time
```

---

## Phase 3: Implement Comparison Methods

### Step 3.1: PCA Reconstruction Baseline
Create file: `src/pca_baseline.py`

```python
from sklearn.decomposition import PCA
import numpy as np
import time

def pca_reconstruction(x_train, x_test, n_components=128):
    """
    Ανακατασκευή εικόνων με PCA για σύγκριση με autoencoder
    (Reconstruct images using PCA for comparison with autoencoder)
    
    Args:
        x_train: Training images
        x_test: Test images
        n_components: Number of PCA components
    
    Returns:
        test_reconstructed, mse, explained_variance, pca_time
    """
    print(f'\n=== PCA Reconstruction with {n_components} components ===')
    
    # Flatten images
    train_flat = x_train.reshape(x_train.shape[0], -1)
    test_flat = x_test.reshape(test_test.shape[0], -1)
    
    # Fit PCA
    start_time = time.time()
    pca = PCA(n_components=n_components)
    pca.fit(train_flat)
    
    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f'Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)')
    
    # Transform and reconstruct
    test_pca = pca.transform(test_flat)
    test_reconstructed = pca.inverse_transform(test_pca)
    
    pca_time = time.time() - start_time
    
    # Reshape back to images
    test_reconstructed = test_reconstructed.reshape(x_test.shape)
    
    # Calculate reconstruction error
    mse = np.mean((x_test - test_reconstructed) ** 2)
    print(f'Reconstruction MSE: {mse:.6f}')
    print(f'PCA time: {pca_time:.2f} seconds')
    
    return test_reconstructed, mse, explained_variance, pca_time
```

---

## Phase 4: Implement Classification Using Latent Space

### Step 4.1: Train Classifier on Latent Features
Create file: `src/classifier.py`

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_latent_classifier(latent_dim=128, num_classes=10):
    """
    Δημιουργία ταξινομητή για το latent space
    (Build classifier for latent space)
    """
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='latent_classifier')
    
    return model


def train_latent_classifier(encoder, x_train, y_train, x_test, y_test, epochs=30, batch_size=128):
    """
    Εκπαίδευση ταξινομητή στο latent space
    (Train classifier on latent space)
    """
    # Extract latent features
    print('\nΕξαγωγή latent features (Extracting latent features)...')
    train_latent = encoder.predict(x_train, batch_size=256, verbose=1)
    test_latent = encoder.predict(x_test, batch_size=256, verbose=1)
    
    # Build and compile classifier
    latent_dim = train_latent.shape[1]
    classifier = build_latent_classifier(latent_dim=latent_dim)
    
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print('\nΕκπαίδευση ταξινομητή (Training classifier)...')
    history = classifier.fit(
        train_latent, y_train,
        validation_data=(test_latent, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = classifier.evaluate(test_latent, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_acc*100:.2f}%')
    
    return classifier, history, test_acc


def nearest_neighbor_classifier(encoder, x_train, y_train, x_test, y_test):
    """
    Ταξινομητής πλησιέστερου γείτονα (Nearest Neighbor classifier)
    """
    print('\n=== Nearest Neighbor Classification ===')
    
    # Extract latent features
    train_latent = encoder.predict(x_train, batch_size=256, verbose=0)
    test_latent = encoder.predict(x_test, batch_size=256, verbose=0)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_latent, y_train)
    
    # Predict
    y_pred = knn.predict(test_latent)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'KNN Accuracy: {accuracy*100:.2f}%')
    
    return accuracy


def nearest_centroid_classifier(encoder, x_train, y_train, x_test, y_test):
    """
    Ταξινομητής πλησιέστερου κέντρου κλάσης (Nearest Class Centroid classifier)
    """
    print('\n=== Nearest Class Centroid Classification ===')
    
    # Extract latent features
    train_latent = encoder.predict(x_train, batch_size=256, verbose=0)
    test_latent = encoder.predict(x_test, batch_size=256, verbose=0)
    
    # Calculate class centroids
    num_classes = len(np.unique(y_train))
    centroids = np.zeros((num_classes, train_latent.shape[1]))
    
    for i in range(num_classes):
        centroids[i] = train_latent[y_train == i].mean(axis=0)
    
    # Classify by nearest centroid
    distances = np.zeros((len(test_latent), num_classes))
    for i in range(num_classes):
        distances[:, i] = np.linalg.norm(test_latent - centroids[i], axis=1)
    
    y_pred = np.argmin(distances, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Nearest Centroid Accuracy: {accuracy*100:.2f}%')
    
    return accuracy
```

---

## Phase 5: Hyperparameter Experiments

### Step 5.1: Experiment with Different Architectures
Create file: `src/experiments.py`

```python
import numpy as np
import pandas as pd
from src.autoencoder import build_autoencoder
from src.train_autoencoder import train_autoencoder
from src.classifier import train_latent_classifier

def run_latent_dim_experiments(x_train, y_train, x_test, y_test):
    """
    Πειράματα με διαφορετικές διαστάσεις latent space
    (Experiments with different latent dimensions)
    """
    latent_dims = [32, 64, 128, 256]
    results = []
    
    for latent_dim in latent_dims:
        print(f'\n{"="*60}')
        print(f'Latent Dimension: {latent_dim}')
        print(f'{"="*60}\n')
        
        # Build and train autoencoder
        autoencoder, encoder, decoder = build_autoencoder(latent_dim=latent_dim)
        history, train_time = train_autoencoder(
            autoencoder, x_train, x_test,
            epochs=50, batch_size=128, lr=1e-3
        )
        
        # Evaluate reconstruction
        test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)[0]
        
        # Train classifier
        classifier, clf_history, test_acc = train_latent_classifier(
            encoder, x_train, y_train, x_test, y_test
        )
        
        # Store results
        results.append({
            'latent_dim': latent_dim,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'test_reconstruction_mse': test_loss,
            'classification_accuracy': test_acc,
            'training_time_seconds': train_time
        })
        
        # Save model
        autoencoder.save(f'results/autoencoder_latent{latent_dim}.h5')
        encoder.save(f'results/encoder_latent{latent_dim}.h5')
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/latent_dim_experiments.csv', index=False)
    print('\n' + '='*60)
    print('Results saved to results/latent_dim_experiments.csv')
    print('='*60)
    
    return df
```

---

## Phase 6: Visualization and Analysis

### Step 6.1: Generate Reconstruction Visualizations
Create file: `src/visualize.py`

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_reconstructions(autoencoder, x_test, y_test, class_names, n_samples=10, save_path='results/reconstructions.png'):
    """
    Οπτικοποίηση ανακατασκευών (Visualize reconstructions)
    """
    # Get random samples
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    original = x_test[indices]
    labels = y_test[indices]
    
    # Reconstruct
    reconstructed = autoencoder.predict(original, verbose=0)
    
    # Denormalize from [-1, 1] to [0, 1]
    original_denorm = (original + 1) / 2
    reconstructed_denorm = (reconstructed + 1) / 2
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original_denorm[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original\n{class_names[labels[i]]}', fontsize=10)
        
        # Reconstructed
        axes[1, i].imshow(np.clip(reconstructed_denorm[i], 0, 1))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_training_curves(history, save_path='results/training_curves.png'):
    """
    Γραφήματα εκπαίδευσης (Training curves)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Autoencoder Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_latent_space_tsne(encoder, x_test, y_test, class_names, save_path='results/latent_tsne.png'):
    """
    t-SNE οπτικοποίηση latent space (t-SNE visualization of latent space)
    """
    print('\nGenerating t-SNE visualization...')
    
    # Extract latent features (use subset for speed)
    n_samples = min(5000, len(x_test))
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    
    latent = encoder.predict(x_test[indices], batch_size=256, verbose=0)
    labels = y_test[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    latent_2d = tsne.fit_transform(latent)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=10)
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)
    
    plt.title('t-SNE Visualization of Latent Space', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def compare_pca_autoencoder(autoencoder, pca_reconstructed, x_test, y_test, class_names, save_path='results/pca_comparison.png'):
    """
    Σύγκριση PCA vs Autoencoder (Compare PCA vs Autoencoder)
    """
    # Get samples
    n_samples = 5
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    original = x_test[indices]
    labels = y_test[indices]
    
    # Autoencoder reconstruction
    ae_reconstructed = autoencoder.predict(original, verbose=0)
    
    # Denormalize
    original_denorm = (original + 1) / 2
    ae_denorm = (ae_reconstructed + 1) / 2
    pca_denorm = (pca_reconstructed[indices] + 1) / 2
    
    # Plot
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original_denorm[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
        axes[0, i].set_title(class_names[labels[i]], fontsize=10)
        
        # PCA
        axes[1, i].imshow(np.clip(pca_denorm[i], 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('PCA', fontsize=12, fontweight='bold')
        
        # Autoencoder
        axes[2, i].imshow(np.clip(ae_denorm[i], 0, 1))
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Autoencoder', fontsize=12, fontweight='bold')
    
    plt.suptitle('Comparison: PCA vs Autoencoder Reconstruction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')
```

---

## Phase 7: Main Execution Script

### Step 7.1: Create Main Script
Create file: `src/main.py`

```python
"""
Κύριο script εκτέλεσης για την 3η Υποχρεωτική Εργασία
(Main execution script for 3rd Mandatory Assignment)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Import custom modules
from load_data import load_cifar10, CLASS_NAMES
from autoencoder import build_autoencoder
from train_autoencoder import train_autoencoder
from pca_baseline import pca_reconstruction
from classifier import (train_latent_classifier, 
                       nearest_neighbor_classifier, 
                       nearest_centroid_classifier)
from visualize import (visualize_reconstructions, 
                      plot_training_curves,
                      plot_latent_space_tsne,
                      compare_pca_autoencoder)
from experiments import run_latent_dim_experiments

def main():
    """
    Κύρια συνάρτηση εκτέλεσης (Main execution function)
    """
    print('\n' + '='*70)
    print('3η ΥΠΟΧΡΕΩΤΙΚΗ ΕΡΓΑΣΙΑ - AUTOENCODER ΓΙΑ CIFAR-10')
    print('3RD MANDATORY ASSIGNMENT - AUTOENCODER FOR CIFAR-10')
    print('='*70 + '\n')
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # ==================== PHASE 1: Load Data ====================
    print('\n[PHASE 1] Φόρτωση δεδομένων (Loading data)...')
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # ==================== PHASE 2: Build & Train Autoencoder ====================
    print('\n[PHASE 2] Δημιουργία και εκπαίδευση Autoencoder (Building and training Autoencoder)...')
    
    LATENT_DIM = 128
    autoencoder, encoder, decoder = build_autoencoder(latent_dim=LATENT_DIM)
    
    # Print model summaries
    print('\n--- Encoder Summary ---')
    encoder.summary()
    print('\n--- Decoder Summary ---')
    decoder.summary()
    print('\n--- Autoencoder Summary ---')
    autoencoder.summary()
    
    # Train
    history, training_time = train_autoencoder(
        autoencoder, x_train, x_test,
        epochs=50, batch_size=128, lr=1e-3
    )
    
    # Plot training curves
    plot_training_curves(history)
    
    # ==================== PHASE 3: PCA Baseline ====================
    print('\n[PHASE 3] PCA Baseline...')
    pca_reconstructed, pca_mse, pca_variance, pca_time = pca_reconstruction(
        x_train, x_test, n_components=LATENT_DIM
    )
    
    # ==================== PHASE 4: Evaluate Reconstruction ====================
    print('\n[PHASE 4] Αξιολόγηση ανακατασκευής (Evaluating reconstruction)...')
    
    # Autoencoder reconstruction error
    ae_mse = autoencoder.evaluate(x_test, x_test, verbose=0)[0]
    print(f'\nAutoencoder Test MSE: {ae_mse:.6f}')
    print(f'PCA Test MSE: {pca_mse:.6f}')
    print(f'Improvement: {((pca_mse - ae_mse) / pca_mse * 100):.2f}%')
    
    # Visualize reconstructions
    visualize_reconstructions(autoencoder, x_test, y_test, CLASS_NAMES)
    compare_pca_autoencoder(autoencoder, pca_reconstructed, x_test, y_test, CLASS_NAMES)
    
    # ==================== PHASE 5: Classification ====================
    print('\n[PHASE 5] Ταξινομοποίηση (Classification)...')
    
    # Train neural network classifier on latent features
    classifier, clf_history, nn_accuracy = train_latent_classifier(
        encoder, x_train, y_train, x_test, y_test
    )
    
    # Nearest Neighbor
    knn_accuracy = nearest_neighbor_classifier(
        encoder, x_train, y_train, x_test, y_test
    )
    
    # Nearest Centroid
    nc_accuracy = nearest_centroid_classifier(
        encoder, x_train, y_train, x_test, y_test
    )
    
    # ==================== PHASE 6: Visualizations ====================
    print('\n[PHASE 6] Οπτικοποιήσεις (Visualizations)...')
    plot_latent_space_tsne(encoder, x_test, y_test, CLASS_NAMES)
    
    # ==================== PHASE 7: Summary ====================
    print('\n' + '='*70)
    print('ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (RESULTS SUMMARY)')
    print('='*70)
    print(f'\nReconstruction:')
    print(f'  - Autoencoder MSE: {ae_mse:.6f}')
    print(f'  - PCA MSE: {pca_mse:.6f}')
    print(f'  - PCA Explained Variance: {pca_variance:.4f}')
    print(f'\nClassification Accuracy:')
    print(f'  - Neural Network: {nn_accuracy*100:.2f}%')
    print(f'  - K-Nearest Neighbors: {knn_accuracy*100:.2f}%')
    print(f'  - Nearest Centroid: {nc_accuracy*100:.2f}%')
    print(f'\nTraining Time:')
    print(f'  - Autoencoder: {training_time:.2f} seconds')
    print(f'  - PCA: {pca_time:.2f} seconds')
    print('='*70 + '\n')
    
    # Save models
    autoencoder.save('results/final_autoencoder.h5')
    encoder.save('results/final_encoder.h5')
    decoder.save('results/final_decoder.h5')
    
    print('\nΌλα τα μοντέλα αποθηκεύτηκαν στο φάκελο results/')
    print('All models saved to results/ directory')

if __name__ == '__main__':
    main()
```

---

## Phase 8: Final Checklist

### Code Requirements
- [ ] Data loading (`src/load_data.py`)
- [ ] Autoencoder model (`src/autoencoder.py`)
- [ ] Training loop (`src/train_autoencoder.py`)
- [ ] PCA baseline (`src/pca_baseline.py`)
- [ ] Classification (`src/classifier.py`)
- [ ] Visualization (`src/visualize.py`)
- [ ] Experiments (`src/experiments.py`)
- [ ] Main script (`src/main.py`)

### Report Requirements
- [ ] Algorithm description (Autoencoder architecture)
- [ ] Correct reconstruction examples
- [ ] Incorrect reconstruction examples
- [ ] Training and testing success rates
- [ ] Training time measurements
- [ ] Results for different latent dimensions
- [ ] Comparison with PCA baseline
- [ ] Comparison with Nearest Neighbor/Nearest Centroid classifiers
- [ ] Code comments in Greek

### Presentation Requirements
- [ ] Slide 1: Student info, datasets, special considerations
- [ ] Slides 2-9: Results for each experiment
- [ ] Slide 10: Conclusions and comments

---

## Quick Start Commands

```bash
# Navigate to project directory
cd "/home/vaggelis/Desktop/AUTH/Neural Networks/Project 3"

# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn pandas tqdm seaborn

# Run main script
python src/main.py

# Run hyperparameter experiments (optional)
python -c "from src.experiments import run_latent_dim_experiments; from src.load_data import load_cifar10; (x_train, y_train), (x_test, y_test) = load_cifar10(); run_latent_dim_experiments(x_train, y_train, x_test, y_test)"
```

---

## Important Notes for LLM Agents

1. **TensorFlow/Keras Built-in Dataset**: CIFAR-10 is built into Keras - no manual download needed
2. **GPU Support**: TensorFlow automatically uses GPU if available (`tf.config.list_physical_devices('GPU')`)
3. **Save Checkpoints**: Models are automatically saved during training via callbacks
4. **Greek Comments**: All code includes Greek comments as per assignment requirements
5. **Normalization**: Images normalized to [-1, 1] to match tanh activation
6. **Results Directory**: All outputs saved to `results/` directory
7. **CSV Logging**: Training metrics automatically logged to CSV files

---

## Expected Timeline

| Phase | Estimated Time | Priority |
|-------|---------------|----------|
| Setup & Data Prep | 30 min | High |
| Autoencoder Implementation | 1-2 hours | High |
| Training (50 epochs) | 1-2 hours (GPU) | High |
| PCA Baseline | 30 min | High |
| Classification | 1 hour | Medium |
| Hyperparameter Experiments | 3-4 hours | Medium |
| Visualization | 1 hour | High |
| Report Writing | 4-6 hours | High |
| **Total** | **12-17 hours** | - |

---

## Success Criteria

✅ **Minimum Requirements**:
- Autoencoder trains successfully and converges
- Reconstruction MSE < 0.1 on test set
- Visual quality of reconstructions is recognizable
- PCA comparison shows autoencoder superiority
- Classification accuracy > 60% using latent features

🎯 **Target Performance**:
- Reconstruction MSE < 0.05
- Classification accuracy > 70% using latent features
- Clear visual quality in reconstructions
- Comprehensive hyperparameter analysis
- Professional report with all required figures
