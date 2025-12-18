import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import load_and_preprocess_data
from models import build_conv_autoencoder

def train_autoencoder(latent_dim=128, epochs=50, batch_size=64, learning_rate=0.001):
    """
    Trains the convolutional autoencoder.
    """
    print(f"\n--- Starting training for Latent Dim: {latent_dim} ---")
    
    # Load data
    (x_train, _), (x_test, _) = load_and_preprocess_data()
    
    # Build model
    autoencoder, encoder, decoder = build_conv_autoencoder(latent_dim=latent_dim)
    
    # Compile model
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                        loss='mse',
                        metrics=['mae'])
    
    # Set up checkpointing
    checkpoint_dir = f"models/latent_{latent_dim}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train
    start_time = time.time()
    history = autoencoder.fit(x_train, x_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(x_test, x_test),
                             verbose=1)
    training_time = time.time() - start_time
    
    # Save models
    autoencoder.save(os.path.join(checkpoint_dir, "autoencoder.keras"))
    encoder.save(os.path.join(checkpoint_dir, "encoder.keras"))
    decoder.save(os.path.join(checkpoint_dir, "decoder.keras"))
    
    # Save history and metrics
    metrics = {
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time": training_time,
        "final_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "history": history.history
    }
    
    metrics_path = f"results/metrics/latent_{latent_dim}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Autoencoder Loss (Latent Dim: {latent_dim})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(f"results/plots/latent_{latent_dim}_loss.png")
    plt.close()
    
    print(f"Training for Latent Dim {latent_dim} completed in {training_time:.2f}s")
    return metrics

if __name__ == "__main__":
    # Example usage
    # We can run this for different latent dimensions as per assignment
    latent_dims = [32, 64, 128, 256]
    for ld in latent_dims:
        train_autoencoder(latent_dim=ld, epochs=20) # Start with 20 epochs for testing
