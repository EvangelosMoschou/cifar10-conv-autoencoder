import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from data_utils import load_and_preprocess_data, get_class_name

def visualize_latent_space(encoder, x_test, y_test, title="Latent Space Visualization"):
    """
    Visualizes the latent space using t-SNE.
    """
    print("Extracting latent features for visualization...")
    latent_feats = encoder.predict(x_test, batch_size=128)
    
    # Subsample for faster t-SNE if needed
    n_samples = min(2000, len(latent_feats))
    indices = np.random.choice(len(latent_feats), n_samples, replace=False)
    latent_subset = latent_feats[indices]
    y_subset = y_test[indices]
    
    print(f"Running t-SNE on {n_samples} samples...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_subset)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label='Classes')
    
    # Add class names to the legend
    classes = [get_class_name(i) for i in range(10)]
    # This is a bit tricky with scatter, let's just label the colorbar or add annotations
    # For simplicity, we'll just note the classes in the title or a separate legend
    
    plt.title(title)
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig("results/plots/latent_space_tsne.png")
    plt.close()
    print("t-SNE plot saved to results/plots/latent_space_tsne.png")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    try:
        encoder = tf.keras.models.load_model("models/latent_128/encoder.keras")
        visualize_latent_space(encoder, x_test, y_test)
    except Exception as e:
        print(f"Could not load encoder for visualization: {e}")
