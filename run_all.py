import os
import sys

# Add src to python path
sys.path.append(os.path.abspath("src"))

from train import train_autoencoder
from pca_comparison import run_pca_reconstruction
from classification import evaluate_classification, get_pca_features, get_ae_features
from visualize_latent import visualize_latent_space
from data_utils import load_and_preprocess_data
import tensorflow as tf
import json

def run_all_experiments():
    """
    Runs all experiments required for the assignment.
    """
    print("=== CIFAR-10 Autoencoder Assignment: Full Experiment Runner ===\n")
    
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 2. Train Autoencoders with different latent dimensions
    latent_dims = [32, 64, 128, 256]
    ae_results = []
    
    for ld in latent_dims:
        metrics = train_autoencoder(latent_dim=ld, epochs=20) # 20 epochs for demonstration
        ae_results.append(metrics)
        
    # 3. PCA Comparison
    print("\n--- Running PCA Comparison ---")
    pca_original, pca_reconstructed, pca_metrics = run_pca_reconstruction(variance_threshold=0.90)
    
    # 4. Classification Evaluation
    classification_results = []
    
    # Raw Pixels
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    res_raw = evaluate_classification(x_train_flat, y_train, x_test_flat, y_test, name="Raw Pixels")
    classification_results.append(res_raw)
    
    # PCA Features
    x_train_pca, x_test_pca, n_comp = get_pca_features(x_train, x_test)
    res_pca = evaluate_classification(x_train_pca, y_train, x_test_pca, y_test, name=f"PCA ({n_comp} comps)")
    classification_results.append(res_pca)
    
    # AE Features (using ld=128 as representative)
    print("\nExtracting AE features for classification (Latent Dim: 128)...")
    encoder_128 = tf.keras.models.load_model("models/latent_128/encoder.keras")
    x_train_ae, x_test_ae = get_ae_features(encoder_128, x_train, x_test)
    res_ae = evaluate_classification(x_train_ae, y_train, x_test_ae, y_test, name="Autoencoder (128)")
    classification_results.append(res_ae)
    
    # 5. Visualization
    print("\n--- Generating Visualizations ---")
    visualize_latent_space(encoder_128, x_test, y_test, title="Latent Space t-SNE (Dim: 128)")
    
    # 6. Summary Report
    all_summary = {
        "autoencoder_metrics": ae_results,
        "pca_metrics": pca_metrics,
        "classification_results": classification_results
    }
    
    with open("results/metrics/summary_report.json", 'w') as f:
        json.dump(all_summary, f, indent=4)
    
    print("\n=== All experiments completed! Results saved to results/ ===")

if __name__ == "__main__":
    run_all_experiments()
