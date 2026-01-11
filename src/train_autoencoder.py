# train_autoencoder.py - Εκπαίδευση με βελτιστοποίηση GPU
"""
Εκπαιδεύει τον ResNet-style Masked Autoencoder για CIFAR-10.
Χρησιμοποιεί tf.data pipeline για μέγιστη απόδοση.
"""

import os
import time
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# GPU Memory Growth - πρέπει να εκτελεστεί ΠΡΙΝ οποιαδήποτε άλλη TensorFlow operation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

# Εισαγωγή τοπικών modules ΜΕΤΑ τη ρύθμιση GPU
from config import (
    LATENT_DIM, MASK_RATIO, PATCH_SIZE, EPOCHS, BATCH_SIZE, 
    LEARNING_RATE, EARLY_STOPPING_PATIENCE, LR_PATIENCE, LR_FACTOR,
    RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, PIXEL_MIN, PIXEL_MAX
)
from data_utils import load_and_preprocess_data, create_mask
from models import build_resnet_mae


# ==================== ΣΥΝΔΥΑΣΜΕΝΗ ΑΠΩΛΕΙΑ ====================
def combined_loss(y_true, y_pred):
    """
    Συνδυασμένη συνάρτηση απώλειας: MSE + SSIM.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = tf.reduce_mean(1.0 - tf.image.ssim(y_true, y_pred, max_val=2.0))
    return mse + 0.5 * ssim_loss


# ==================== ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ====================
def prepare_dataset(x_data, batch_size, mask_ratio, patch_size, training=True):
    """
    Δημιουργεί tf.data dataset με masking.
    """
    @tf.function
    def apply_augment_and_mask(image):
        # Επαύξηση δεδομένων (μόνο κατά την εκπαίδευση)
        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, PIXEL_MIN, PIXEL_MAX)
        
        original = image
        
        # Δημιουργία μάσκας
        batch_img = tf.expand_dims(image, 0)
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        
        h_patches = height // patch_size
        w_patches = width // patch_size
        num_patches = h_patches * w_patches
        num_masked = tf.cast(tf.cast(num_patches, tf.float32) * mask_ratio, tf.int32)
        
        indices = tf.random.shuffle(tf.range(num_patches))[:num_masked]
        mask = tf.ones((num_patches,), dtype=tf.float32)
        updates = tf.zeros((num_masked,), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.reshape(indices, (-1, 1)), updates)
        mask = tf.reshape(mask, (h_patches, w_patches, 1))
        mask = tf.image.resize(mask, (height, width), method='nearest')
        
        masked_image = image * mask
        
        return masked_image, original

    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    
    if training:
        dataset = dataset.shuffle(buffer_size=10000)
    
    dataset = dataset.map(apply_augment_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ==================== MAIN TRAINING FUNCTION ====================
def train(epochs=EPOCHS, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM, 
          mask_ratio=MASK_RATIO, patch_size=PATCH_SIZE, learning_rate=LEARNING_RATE):
    """
    Κύρια συνάρτηση εκπαίδευσης.
    """
    print("\n" + "="*60)
    print(f"  MASKED AUTOENCODER TRAINING")
    print(f"  latent_dim={latent_dim}, mask_ratio={mask_ratio}, patch_size={patch_size}")
    print("="*60 + "\n")
    
    # Δημιουργία φακέλων
    for subdir in [MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        os.makedirs(subdir, exist_ok=True)

    # Φόρτωση δεδομένων
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build autoencoder
    autoencoder, encoder, decoder = build_resnet_mae(latent_dim=latent_dim)
    
    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=LR_FACTOR, 
            patience=LR_PATIENCE,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{MODELS_DIR}/best_autoencoder.h5', 
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(f'{METRICS_DIR}/training_log.csv')
    ]
    
    # Create datasets
    print("Creating data pipelines...")
    train_dataset = prepare_dataset(x_train, batch_size, mask_ratio, patch_size, training=True)
    val_dataset = prepare_dataset(x_test, batch_size, mask_ratio, patch_size, training=False)
    
    # Training
    print(f"Εκπαίδευση για {epochs} εποχές...")
    start_time = time.time()
    
    history = autoencoder.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nΕκπαίδευση ολοκληρώθηκε σε {training_time/60:.1f} λεπτά")
    
    # Αποθήκευση μοντέλων
    autoencoder.save(f'{MODELS_DIR}/final_autoencoder.h5')
    encoder.save(f'{MODELS_DIR}/final_encoder.h5')
    decoder.save(f'{MODELS_DIR}/final_decoder.h5')
    print(f"Μοντέλα αποθηκεύτηκαν στο {MODELS_DIR}/")
    
    # Γραφική απεικόνιση
    plot_training_history(history)
    
    return history, autoencoder, encoder, decoder


def plot_training_history(history):
    """
    Αποθηκεύει γράφημα με την εξέλιξη της απώλειας.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Απώλεια Εκπαίδευσης', linewidth=2)
    plt.plot(history.history['val_loss'], label='Απώλεια Επικύρωσης', linewidth=2)
    plt.title('Εξέλιξη Απώλειας Κατά την Εκπαίδευση', fontsize=14)
    plt.xlabel('Εποχή', fontsize=12)
    plt.ylabel('Απώλεια (MSE + SSIM)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/training_loss.png', dpi=150)
    plt.close()
    print(f"Γράφημα αποθηκεύτηκε στο {PLOTS_DIR}/training_loss.png")


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Masked Autoencoder on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Latent dimension')
    parser.add_argument('--mask_ratio', type=float, default=MASK_RATIO, help='Mask ratio')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        mask_ratio=args.mask_ratio,
        learning_rate=args.lr
    )
