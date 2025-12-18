import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd

from data_utils import load_and_preprocess_data, plot_reconstructions, get_class_name
from models import build_autoencoder

def evaluate_models():
    # Create subdirectories if they don't exist
    for subdir in ['plots', 'metrics']:
        os.makedirs(os.path.join('results', subdir), exist_ok=True)

    # Load Data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Load Models
    try:
        autoencoder = keras.models.load_model('results/models/final_autoencoder.h5', compile=False)
        encoder = keras.models.load_model('results/models/final_encoder.h5', compile=False)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Could not load models. Error: {e}")
        return

    # 1. Reconstruction Evaluation
    print("\n--- Reconstruction Evaluation ---")
    reconstructed_images = autoencoder.predict(x_test)
    mse = np.mean(np.square(x_test - reconstructed_images))
    print(f"Autoencoder MSE on Test Set: {mse:.6f}")
    
    # Denormalize for plotting: [-1, 1] -> [0, 1]
    x_test_plot = (x_test + 1) / 2
    reconstructed_plot = (reconstructed_images + 1) / 2
    
    # Plot Reconstructions
    plot_reconstructions(x_test_plot, reconstructed_plot, title="Autoencoder Reconstructions")
    plt.savefig('results/plots/reconstructions.png')
    plt.close()

    # 2. PCA Comparison
    print("\n--- PCA Comparison ---")
    latent_dim = encoder.output_shape[1]
    print(f"Fitting PCA with {latent_dim} components...")
    
    # Flatten data for PCA
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))
    
    pca = PCA(n_components=latent_dim)
    pca.fit(x_train_flat)
    
    x_test_pca = pca.transform(x_test_flat)
    x_test_pca_recon = pca.inverse_transform(x_test_pca)
    
    pca_mse =mean_squared_error(x_test_flat, x_test_pca_recon)
    print(f"PCA MSE on Test Set: {pca_mse:.6f}")
    
    # Plot PCA Reconstructions
    x_test_pca_recon_img = x_test_pca_recon.reshape(x_test.shape)
    x_test_pca_recon_plot = (x_test_pca_recon_img + 1) / 2
    
    plot_reconstructions(x_test_plot, x_test_pca_recon_plot, title=f"PCA Reconstructions (n={latent_dim})")
    plt.savefig('results/plots/pca_comparison.png')
    plt.close()

    # 3. Latent Space Classification
    print("\n--- Latent Space Classification ---")
    
    # Get Latent Representations
    z_train = encoder.predict(x_train)
    z_test = encoder.predict(x_test)
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(z_train, y_train)
    knn_acc = knn.score(z_test, y_test)
    print(f"KNN Accuracy on Latent Space: {knn_acc:.4f}")
    
    # Nearest Centroid
    nc = NearestCentroid()
    nc.fit(z_train, y_train)
    nc_acc = nc.score(z_test, y_test)
    print(f"Nearest Centroid Accuracy on Latent Space: {nc_acc:.4f}")
    
    # Neural Network Classifier
    print("Training Neural Network on Latent Space...")
    classifier = keras.Sequential([
        keras.layers.Input(shape=(latent_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf_history = classifier.fit(z_train, y_train, epochs=20, batch_size=128, validation_data=(z_test, y_test), verbose=0)
    
    nn_loss, nn_acc = classifier.evaluate(z_test, y_test, verbose=0)
    print(f"Neural Network Accuracy on Latent Space: {nn_acc:.4f}")
    
    # Plot Classifier Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(clf_history.history['accuracy'], label='Train Accuracy')
    plt.plot(clf_history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Classifier Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(clf_history.history['loss'], label='Train Loss')
    plt.plot(clf_history.history['val_loss'], label='Val Loss')
    plt.title('Classifier Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/classifier_history.png')
    plt.close()
    print("Classifier history plot saved to results/plots/classifier_history.png")
    
    # Save Metrics
    results = {
        'Autoencoder_MSE': mse,
        'PCA_MSE': pca_mse,
        'KNN_Accuracy': knn_acc,
        'NC_Accuracy': nc_acc,
        'NN_Accuracy': nn_acc
    }
    pd.DataFrame([results]).to_csv('results/metrics/evaluation_metrics.csv', index=False)
    print("Evaluation Complete. Metrics saved to results/metrics/evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_models()
