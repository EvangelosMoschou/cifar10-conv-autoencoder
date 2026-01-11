import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from data_utils import load_and_preprocess_data
from sklearn.decomposition import PCA

def evaluate_classification(x_train_feats, y_train, x_test_feats, y_test, name="Χαρακτηριστικά"):
    """
    Αξιολογεί KNN και NCC στα δεδομένα χαρακτηριστικά.
    """
    print(f"\nΑξιολόγηση Ταξινόμησης για: {name}")
    
    # KNN (Πλησιέστερος Γείτονας)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_feats, y_train)
    y_pred_knn = knn.predict(x_test_feats)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    
    # NCC (Πλησιέστερο Κεντροειδές Κλάσης)
    ncc = NearestCentroid()
    ncc.fit(x_train_feats, y_train)
    y_pred_ncc = ncc.predict(x_test_feats)
    acc_ncc = accuracy_score(y_test, y_pred_ncc)
    
    print(f"  Ακρίβεια KNN: {acc_knn:.4f}")
    print(f"  Ακρίβεια NCC: {acc_ncc:.4f}")
    
    return {
        "name": name,
        "knn_acc": float(acc_knn),
        "ncc_acc": float(acc_ncc)
    }

def get_pca_features(x_train, x_test, variance_threshold=0.90):
    """
    Εξάγει χαρακτηριστικά PCA με το καθορισμένο όριο διακύμανσης.
    """
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    pca = PCA(n_components=variance_threshold)
    x_train_pca = pca.fit_transform(x_train_flat)
    x_test_pca = pca.transform(x_test_flat)
    
    return x_train_pca, x_test_pca, pca.n_components_

def get_ae_features(encoder, x_train, x_test):
    """
    Εξάγει χαρακτηριστικά λανθάνοντος χώρου χρησιμοποιώντας τον κωδικοποιητή.
    """
    x_train_ae = encoder.predict(x_train, batch_size=128)
    x_test_ae = encoder.predict(x_test, batch_size=128)
    
    return x_train_ae, x_test_ae

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Ακατέργαστα Pixels (υποδειγματοληψία για ταχύτητα αν χρειαστεί)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    evaluate_classification(x_train_flat, y_train, x_test_flat, y_test, name="Ακατέργαστα Pixels")
    
    # Χαρακτηριστικά PCA
    x_train_pca, x_test_pca, n_comp = get_pca_features(x_train, x_test)
    evaluate_classification(x_train_pca, y_train, x_test_pca, y_test, name=f"PCA ({n_comp} συνιστώσες)")
    
    # Χαρακτηριστικά Αυτοκωδικοποιητή (παράδειγμα για latent_dim=128)
    try:
        encoder = tf.keras.models.load_model("models/latent_128/encoder.keras")
        x_train_ae, x_test_ae = get_ae_features(encoder, x_train, x_test)
        evaluate_classification(x_train_ae, y_train, x_test_ae, y_test, name="Αυτοκωδικοποιητής (Dim: 128)")
    except Exception as e:
        print(f"Αδυναμία φόρτωσης μοντέλου AE για ταξινόμηση: {e}")
