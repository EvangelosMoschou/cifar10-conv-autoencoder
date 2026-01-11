# data_utils.py - Utilities για δεδομένα CIFAR-10
"""
Περιέχει συναρτήσεις για φόρτωση, προεπεξεργασία και οπτικοποίηση δεδομένων.
Χρησιμοποιεί τις κεντρικές ρυθμίσεις από το config.py.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

from config import (
    PATCH_SIZE, MASK_RATIO, PIXEL_MIN, PIXEL_MAX, CLASS_NAMES
)


def load_and_preprocess_data():
    """
    Φορτώνει τα δεδομένα CIFAR-10 και τα κανονικοποιεί στο εύρος [-1, 1].
    
    Returns:
        (x_train, y_train), (x_test, y_test): Τα δεδομένα εκπαίδευσης και ελέγχου
    """
    print("Φόρτωση δεδομένων CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Κανονικοποίηση τιμών pixel: [0, 255] -> [-1, 1]
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    
    # Επιπεδοποίηση ετικετών
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Train data: {x_train.shape}, range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Test data: {x_test.shape}, range: [{x_test.min():.2f}, {x_test.max():.2f}]")
    
    return (x_train, y_train), (x_test, y_test)


def create_mask(images, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO):
    """
    Δημιουργεί μάσκα που μηδενίζει τυχαία patches της εικόνας.
    
    Args:
        images: Tensor εικόνων (batch_size, height, width, channels)
        patch_size: Μέγεθος κάθε patch (default από config)
        mask_ratio: Ποσοστό patches προς masking (default από config)
    
    Returns:
        mask: Binary mask tensor (batch_size, height, width, 1)
    """
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    
    h_patches = height // patch_size
    w_patches = width // patch_size
    num_patches = h_patches * w_patches
    num_masked = tf.cast(tf.cast(num_patches, tf.float32) * mask_ratio, tf.int32)
    
    # Τυχαία επιλογή patches για masking
    mask_indices = tf.random.shuffle(tf.range(num_patches))[:num_masked]
    
    # Δημιουργία binary mask
    mask = tf.ones((num_patches,), dtype=tf.float32)
    indices = tf.reshape(mask_indices, (-1, 1))
    updates = tf.zeros((num_masked,), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(mask, indices, updates)
    
    # Αναδιαμόρφωση και upscaling
    mask = tf.reshape(mask, (h_patches, w_patches, 1))
    mask = tf.image.resize(mask, (height, width), method='nearest')
    
    return mask


def augment_image(image):
    """
    Εφαρμόζει τυχαία επαύξηση δεδομένων σε μία εικόνα.
    
    Args:
        image: Tensor εικόνας (height, width, channels)
    
    Returns:
        augmented: Επαυξημένη εικόνα
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, PIXEL_MIN, PIXEL_MAX)
    return image


def plot_reconstructions(original, reconstructed, masked=None, n=10, title="Ανακατασκευές"):
    """
    Οπτικοποίηση αρχικών εναντίον ανακατασκευασμένων εικόνων.
    
    Args:
        original: Αρχικές εικόνες
        reconstructed: Ανακατασκευασμένες εικόνες
        masked: Εικόνες με μάσκα (προαιρετικό)
        n: Αριθμός εικόνων προς εμφάνιση
        title: Τίτλος γραφήματος
    """
    rows = 3 if masked is not None else 2
    fig, axes = plt.subplots(rows, n, figsize=(2*n, 2*rows))
    
    for i in range(n):
        # Αρχική εικόνα
        ax = axes[0, i] if rows > 1 else axes[i]
        img = np.clip(original[i], 0, 1)  # Clip to valid range
        ax.imshow(img)
        if i == 0:
            ax.set_title("Αρχική", fontsize=10)
        ax.axis('off')
        
        # Εικόνα με μάσκα
        if masked is not None:
            ax = axes[1, i]
            img = np.clip(masked[i], 0, 1)
            ax.imshow(img)
            if i == 0:
                ax.set_title("Με Μάσκα", fontsize=10)
            ax.axis('off')
        
        # Ανακατασκευή
        ax = axes[rows-1, i]
        img = np.clip(reconstructed[i], 0, 1)
        ax.imshow(img)
        if i == 0:
            ax.set_title("Ανακατασκευή", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def denormalize(images):
    """
    Μετατρέπει εικόνες από [-1, 1] σε [0, 1] για οπτικοποίηση.
    
    Args:
        images: Tensor εικόνων στο εύρος [-1, 1]
    
    Returns:
        images: Tensor εικόνων στο εύρος [0, 1]
    """
    return (images + 1.0) / 2.0


def get_class_name(label):
    """
    Επιστρέφει το όνομα κλάσης για δεδομένη ετικέτα CIFAR-10.
    
    Args:
        label: Integer ετικέτα (0-9)
    
    Returns:
        name: Όνομα κλάσης (string)
    """
    return CLASS_NAMES[label]
