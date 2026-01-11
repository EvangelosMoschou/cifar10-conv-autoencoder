import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from data_utils import load_and_preprocess_data, get_class_name

def visualize_latent_space(encoder, x_test, y_test, title="Οπτικοποίηση Λανθάνοντος Χώρου"):
    """
    Οπτικοποιεί τον λανθάνοντα χώρο χρησιμοποιώντας t-SNE.
    """
    print("Εξαγωγή χαρακτηριστικών λανθάνοντος χώρου για οπτικοποίηση...")
    latent_feats = encoder.predict(x_test, batch_size=128)
    
    # Υποδειγματοληψία για ταχύτερο t-SNE αν χρειαστεί
    n_samples = min(2000, len(latent_feats))
    indices = np.random.choice(len(latent_feats), n_samples, replace=False)
    latent_subset = latent_feats[indices]
    y_subset = y_test[indices]
    
    print(f"Εκτέλεση t-SNE σε {n_samples} δείγματα...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_subset)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label='Classes')
    
    # Προσθήκη ονομάτων κλάσεων στο υπόμνημα
    classes = [get_class_name(i) for i in range(10)]
    # Αυτό είναι λίγο δύσκολο με scatter, ας σημειώσουμε τις κλάσεις στον τίτλο ή σε ξεχωριστό υπόμνημα
    # Για απλότητα, θα σημειώσουμε τις κλάσεις στον τίτλο
    
    plt.title(title)
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig("results/plots/latent_space_tsne.png")
    plt.close()
    print("Γράφημα t-SNE αποθηκεύτηκε στο results/plots/latent_space_tsne.png")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    try:
        encoder = tf.keras.models.load_model("models/latent_128/encoder.keras")
        visualize_latent_space(encoder, x_test, y_test)
    except Exception as e:
        print(f"Αδυναμία φόρτωσης κωδικοποιητή για οπτικοποίηση: {e}")
