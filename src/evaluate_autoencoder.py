# evaluate_autoencoder.py - Αξιολόγηση και Σύγκριση Autoencoder
"""
Πλήρης αξιολόγηση του εκπαιδευμένου Autoencoder:
1. Αξιολόγηση ανακατασκευής (MSE, SSIM)
2. Σύγκριση με PCA
3. Ταξινόμηση στον Λανθάνοντα Χώρο (KNN, NC, MLP)
4. Fine-tuning για ταξινόμηση
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import mean_squared_error
import pandas as pd

# GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from config import (
    LATENT_DIM, MASK_RATIO, PATCH_SIZE, 
    MODELS_DIR, PLOTS_DIR, METRICS_DIR
)
from data_utils import load_and_preprocess_data, create_mask, denormalize, plot_reconstructions


def evaluate_models():
    """
    Πλήρης αξιολόγηση του εκπαιδευμένου autoencoder.
    """
    print("\n" + "="*60)
    print("  AUTOENCODER EVALUATION")
    print("="*60 + "\n")
    
    # Δημιουργία φακέλων
    for subdir in [PLOTS_DIR, METRICS_DIR]:
        os.makedirs(subdir, exist_ok=True)

    # Φόρτωση δεδομένων
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Φόρτωση μοντέλων
    try:
        autoencoder = keras.models.load_model(f'{MODELS_DIR}/final_autoencoder.h5', compile=False)
        encoder = keras.models.load_model(f'{MODELS_DIR}/final_encoder.h5', compile=False)
        print("✓ Μοντέλα φορτώθηκαν επιτυχώς")
    except Exception as e:
        print(f"✗ Αδυναμία φόρτωσης μοντέλων: {e}")
        print("  Εκτελέστε πρώτα το train_autoencoder.py")
        return None
    
    latent_dim = encoder.output_shape[1]
    print(f"  Latent dimension: {latent_dim}")
    
    results = {}
    
    # ==================== 1. RECONSTRUCTION EVALUATION ====================
    print("\n--- 1. Αξιολόγηση Ανακατασκευής ---")
    
    # Αξιολόγηση χωρίς μάσκα (clean reconstruction)
    reconstructed_clean = autoencoder.predict(x_test, verbose=0)
    mse_clean = np.mean(np.square(x_test - reconstructed_clean))
    print(f"MSE (χωρίς μάσκα): {mse_clean:.6f}")
    results['AE_MSE_clean'] = mse_clean
    
    # Αξιολόγηση με μάσκα (masked reconstruction)
    mask = create_mask(x_test[:100], patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
    masked_x_test = x_test[:100] * mask.numpy()
    reconstructed_masked = autoencoder.predict(masked_x_test, verbose=0)
    mse_masked = np.mean(np.square(x_test[:100] - reconstructed_masked))
    print(f"MSE (με μάσκα {MASK_RATIO*100:.0f}%): {mse_masked:.6f}")
    results['AE_MSE_masked'] = mse_masked
    
    # Οπτικοποίηση ανακατασκευών
    fig = plot_reconstructions(
        denormalize(x_test[:10]), 
        denormalize(reconstructed_clean[:10]),
        masked=denormalize(masked_x_test[:10]),
        title="Ανακατασκευές Autoencoder"
    )
    plt.savefig(f'{PLOTS_DIR}/reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Γράφημα: {PLOTS_DIR}/reconstructions.png")
    
    # ==================== 2. PCA COMPARISON ====================
    print("\n--- 2. Σύγκριση με PCA ---")
    
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))
    
    pca = PCA(n_components=latent_dim)
    pca.fit(x_train_flat)
    
    x_test_pca = pca.transform(x_test_flat)
    x_test_pca_recon = pca.inverse_transform(x_test_pca)
    
    pca_mse = mean_squared_error(x_test_flat, x_test_pca_recon)
    print(f"PCA MSE (n={latent_dim}): {pca_mse:.6f}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    results['PCA_MSE'] = pca_mse
    
    # Σύγκριση
    improvement = (pca_mse - mse_clean) / pca_mse * 100
    print(f"\nAutoencoder vs PCA: {improvement:+.1f}% {'καλύτερο' if improvement > 0 else 'χειρότερο'}")
    
    # Οπτικοποίηση PCA
    x_test_pca_recon_img = x_test_pca_recon.reshape(x_test.shape)
    fig = plot_reconstructions(
        denormalize(x_test[:10]), 
        denormalize(x_test_pca_recon_img[:10]),
        title=f"Ανακατασκευές PCA (n={latent_dim})"
    )
    plt.savefig(f'{PLOTS_DIR}/pca_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Γράφημα: {PLOTS_DIR}/pca_comparison.png")
    
    # ==================== 3. CLASSIFICATION ====================
    print("\n--- 3. Ταξινόμηση στον Λανθάνοντα Χώρο ---")
    
    # Extract latent representations
    z_train = encoder.predict(x_train, verbose=0)
    z_test = encoder.predict(x_test, verbose=0)
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(z_train, y_train)
    knn_acc = knn.score(z_test, y_test)
    print(f"KNN (k=5): {knn_acc*100:.2f}%")
    results['KNN_Accuracy'] = knn_acc
    
    # Nearest Centroid
    nc = NearestCentroid()
    nc.fit(z_train, y_train)
    nc_acc = nc.score(z_test, y_test)
    print(f"Nearest Centroid: {nc_acc*100:.2f}%")
    results['NC_Accuracy'] = nc_acc
    
    # Neural Network Classifier (frozen encoder)
    print("\nΕκπαίδευση NN ταξινομητή (frozen encoder)...")
    classifier = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    classifier.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    classifier_history = classifier.fit(
        z_train, y_train,
        epochs=30,
        batch_size=128,
        validation_data=(z_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=0
    )
    
    _, nn_acc = classifier.evaluate(z_test, y_test, verbose=0)
    print(f"NN Classifier (frozen): {nn_acc*100:.2f}%")
    results['NN_Accuracy'] = nn_acc
    
    # Plot Frozen Classifier History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(classifier_history.history['accuracy'], label='Train')
    plt.plot(classifier_history.history['val_accuracy'], label='Validation')
    plt.title('Frozen Classifier Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(classifier_history.history['loss'], label='Train')
    plt.plot(classifier_history.history['val_loss'], label='Validation')
    plt.title('Frozen Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/frozen_classifier_history.png', dpi=150)
    plt.close()
    results['NN_Accuracy'] = nn_acc
    
    # ==================== 4. FINE-TUNING ====================
    print("\n--- 4. Fine-tuning End-to-End ---")
    
    encoder.trainable = True
    
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomContrast(0.1)(x)
    
    # Encode
    encoded = encoder(x)
    
    # Classifier head with regularization
    l2_reg = keras.regularizers.l2(1e-4)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    finetuned_model = keras.Model(inputs, outputs)
    
    finetuned_model.compile(
        optimizer=keras.optimizers.Adam(1e-5, clipnorm=1.0),  # Much lower LR and clipping
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = finetuned_model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=1
    )
    
    _, ft_acc = finetuned_model.evaluate(x_test, y_test, verbose=0)
    print(f"Fine-tuned Accuracy: {ft_acc*100:.2f}%")
    results['FineTuned_Accuracy'] = ft_acc
    
    # Plot fine-tuning history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Fine-tuning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/finetuning_history.png', dpi=150)
    plt.close()
    
    # ==================== SAVE RESULTS ====================
    print("\n--- Αποτελέσματα ---")
    print("="*40)
    print(f"{'Μέθοδος':<25} {'Τιμή':>10}")
    print("-"*40)
    for key, value in results.items():
        if 'Accuracy' in key:
            print(f"{key:<25} {value*100:>9.2f}%")
        else:
            print(f"{key:<25} {value:>10.6f}")
    print("="*40)
    
    # Save to CSV
    pd.DataFrame([results]).to_csv(f'{METRICS_DIR}/evaluation_metrics.csv', index=False)
    print(f"\nΜετρικές αποθηκεύτηκαν στο {METRICS_DIR}/evaluation_metrics.csv")
    
    return results


if __name__ == "__main__":
    evaluate_models()
