# train.py - Εκπαίδευση για πολλαπλές διαστάσεις latent space
"""
Εκπαιδεύει τον autoencoder με διαφορετικές τιμές latent_dim
για ablation study και σύγκριση απόδοσης.
"""

import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from config import (
    LATENT_DIM, MASK_RATIO, PATCH_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR
)
from data_utils import load_and_preprocess_data, create_mask
from models import build_conv_autoencoder


def train_autoencoder(latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                      learning_rate=LEARNING_RATE, mask_ratio=MASK_RATIO, patch_size=PATCH_SIZE):
    """
    Εκπαιδεύει τον συνελικτικό αυτοκωδικοποιητή.
    
    Args:
        latent_dim: Διάσταση bottleneck
        epochs: Αριθμός εποχών
        batch_size: Μέγεθος batch
        learning_rate: Learning rate
        mask_ratio: Ποσοστό masking
        patch_size: Μέγεθος patches
    
    Returns:
        metrics: Dictionary με μετρικές εκπαίδευσης
    """
    print(f"\n{'='*60}")
    print(f"  Training Autoencoder (latent_dim={latent_dim})")
    print(f"{'='*60}\n")
    
    # Δημιουργία φακέλων
    checkpoint_dir = f"{RESULTS_DIR}/models/latent_{latent_dim}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Φόρτωση δεδομένων
    (x_train, _), (x_test, _) = load_and_preprocess_data()
    
    # Κατασκευή μοντέλου
    autoencoder, encoder, decoder = build_conv_autoencoder(latent_dim=latent_dim)
    
    # Compile
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='mse',
        metrics=['mae']
    )
    
    # tf.data pipeline
    def apply_mask(image):
        mask = create_mask(tf.expand_dims(image, 0), patch_size=patch_size, mask_ratio=mask_ratio)
        masked_image = image * tf.squeeze(mask, 0)
        return masked_image, image
    
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(buffer_size=10000)
        .map(apply_mask, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(x_test)
        .map(apply_mask, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', verbose=1),
        ModelCheckpoint(f"{checkpoint_dir}/best_autoencoder.keras", save_best_only=True, 
                       monitor='val_loss', verbose=1)
    ]
    
    # Εκπαίδευση
    start_time = time.time()
    history = autoencoder.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Αποθήκευση μοντέλων
    autoencoder.save(f"{checkpoint_dir}/autoencoder.keras")
    encoder.save(f"{checkpoint_dir}/encoder.keras")
    decoder.save(f"{checkpoint_dir}/decoder.keras")
    print(f"Μοντέλα αποθηκεύτηκαν στο {checkpoint_dir}/")
    
    # Αποθήκευση μετρικών
    metrics = {
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mask_ratio": mask_ratio,
        "patch_size": patch_size,
        "training_time": training_time,
        "final_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
    }
    
    with open(f"{METRICS_DIR}/latent_{latent_dim}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Γραφική απεικόνιση
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'Autoencoder Loss (latent_dim={latent_dim})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/latent_{latent_dim}_loss.png", dpi=150)
    plt.close()
    
    print(f"\nΕκπαίδευση ολοκληρώθηκε σε {training_time/60:.1f} λεπτά")
    print(f"Final Loss: {metrics['final_loss']:.6f}, Val Loss: {metrics['final_val_loss']:.6f}")
    
    return metrics


if __name__ == "__main__":
    # Ablation study: εκπαίδευση για διαφορετικές τιμές latent_dim
    latent_dims = [32, 64, 128, 256]
    all_metrics = []
    
    for ld in latent_dims:
        metrics = train_autoencoder(latent_dim=ld, epochs=30)
        all_metrics.append(metrics)
    
    # Summary
    print("\n" + "="*60)
    print("  ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"{'Latent Dim':<12} {'Train Loss':<12} {'Val Loss':<12} {'Time (min)':<10}")
    print("-"*46)
    for m in all_metrics:
        print(f"{m['latent_dim']:<12} {m['final_loss']:<12.6f} {m['final_val_loss']:<12.6f} {m['training_time']/60:<10.1f}")
