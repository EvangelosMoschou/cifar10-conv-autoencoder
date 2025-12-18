import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse

from data_utils import load_and_preprocess_data
from models import build_autoencoder

def train(epochs=50, batch_size=128, latent_dim=128):
    # Create subdirectories
    for subdir in ['models', 'metrics']:
        os.makedirs(os.path.join('results', subdir), exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Build model
    autoencoder, encoder, decoder = build_autoencoder(latent_dim=latent_dim)
    
    # Compile
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('results/models/best_autoencoder.h5', monitor='val_loss', save_best_only=True),
        keras.callbacks.CSVLogger('results/metrics/training_log.csv')
    ]

    # Train
    print(f"Starting training for {epochs} epochs...")
    history = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=callbacks
    )

    # Save final models
    autoencoder.save('results/models/final_autoencoder.h5')
    encoder.save('results/models/final_encoder.h5')
    decoder.save('results/models/final_decoder.h5')
    
    print("Training complete.")
    
    # Plot Loss
    plot_training_loss(history)
    
    return history

def plot_training_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/training_loss.png')
    plt.close()
    print("Training loss plot saved to results/plots/training_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, latent_dim=args.latent_dim)
