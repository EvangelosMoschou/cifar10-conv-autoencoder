import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from data_utils import load_and_preprocess_data

def run_pca_reconstruction(n_components=None, variance_threshold=0.90):
    """
    Εκτελεί ανακατασκευή PCA στα δεδομένα CIFAR-10.
    """
    (x_train, _), (x_test, _) = load_and_preprocess_data()
    
    # Επιπεδοποίηση εικόνων για PCA
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    if n_components is None:
        print(f"Εύρεση συνιστωσών για >= {variance_threshold*100}% διακύμανση...")
        pca = PCA(n_components=variance_threshold)
        pca.fit(x_train_flat)
        n_components = pca.n_components_
        print(f"Επιλέχθηκαν {n_components} συνιστώσες.")
    else:
        pca = PCA(n_components=n_components)
        pca.fit(x_train_flat)
        
    # Μετασχηματισμός και ανακατασκευή
    x_test_pca = pca.transform(x_test_flat)
    x_test_reconstructed_flat = pca.inverse_transform(x_test_pca)
    
    # Αναδιαμόρφωση σε εικόνες
    x_test_reconstructed = x_test_reconstructed_flat.reshape(x_test.shape)
    
    # Περιορισμός τιμών στο [0, 1]
    x_test_reconstructed = np.clip(x_test_reconstructed, 0, 1)
    
    # Υπολογισμός μετρικών
    mse = np.mean(np.square(x_test - x_test_reconstructed))
    
    # Υπολογισμός SSIM (μέσος όρος σε όλες τις εικόνες ελέγχου)
    ssim_val = 0
    for i in range(len(x_test)):
        ssim_val += ssim(x_test[i], x_test_reconstructed[i], channel_axis=2, data_range=1.0)
    ssim_val /= len(x_test)
    
    print(f"Μετρικές Ανακατασκευής PCA (n_components={n_components}):")
    print(f"MSE: {mse:.6f}")
    print(f"SSIM: {ssim_val:.6f}")
    
    return x_test, x_test_reconstructed, {
        "n_components": int(n_components),
        "mse": float(mse),
        "ssim": float(ssim_val)
    }

if __name__ == "__main__":
    original, reconstructed, metrics = run_pca_reconstruction()
    # Αποθήκευση μερικών παραδειγμάτων
    plt.figure(figsize=(20, 4))
    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(original[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(reconstructed[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("results/plots/pca_reconstruction_examples.png")
    plt.close()
