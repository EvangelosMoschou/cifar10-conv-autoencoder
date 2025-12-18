import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data():
    """
    Loads CIFAR-10 data and normalizes it to [0, 1].
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to be between -1 and 1
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    
    # Flatten y labels for easier use in some classifiers
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Train data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def plot_reconstructions(original, reconstructed, n=10, title="Reconstructions"):
    """
    Plots original vs reconstructed images.
    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.suptitle(title)
    plt.show()

def get_class_name(label):
    """
    Returns the class name for a given CIFAR-10 label.
    """
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    return classes[label]
