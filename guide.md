# detailed Guide: Autoencoder for CIFAR-10

This guide explains the technical details, architecture choices, and theoretical background of the Autoencoder implementation for the CIFAR-10 dataset.

## 1. Introduction to Autoencoders

An **Autoencoder** is a type of neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction.

The network consists of two main parts:
1.  **Encoder**: Maps the input $x$ to a latent representation $z$. ($z = f(x)$)
2.  **Decoder**: Maps the latent representation $z$ back to a reconstruction $\hat{x}$. ($\hat{x} = g(z)$)

The goal is to minimize the reconstruction error $\mathcal{L}(x, \hat{x})$, typically Mean Squared Error (MSE).

## 2. Architecture

We implement a **Convolutional Autoencoder** because we are dealing with image data (CIFAR-10). Convolutional layers are better suited for capturing spatial hierarchies in images compared to fully connected layers.

### Encoder
The encoder compresses the 32x32x3 input image into a lower-dimensional latent vector.

-   **Input**: 32x32x3 Image
-   **Conv2D Layers**: Apply filters to extract features. We use strided convolutions (stride=2) instead of pooling layers for downsampling.
    -   Activation: ReLU
    -   Batch Normalization: To accelerate training and stabilize learning.
-   **Flatten**: Converts the 2D feature maps into a 1D vector.
-   **Dense (Latent Space)**: A fully connected layer that outputs the code $z$. The size of this layer (e.g., 128) determines the varying compression ratio.

### Decoder
The decoder attempts to reconstruct the original image from the latent vector.

-   **Input**: Latent vector $z$
-   **Dense**: Reshapes the vector back to the spatial dimensions of the last encoder feature map.
-   **Conv2DTranspose Layers**: Also known as "deconvolutions", these layers upsample the feature maps back to the original resolution.
-   **Output Layer**: A final Conv2D layer with `tanh` activation.
    -   **Why `tanh`?** We normalize our input images to the range [-1, 1]. `tanh` outputs values in this exact range, making it the natural choice for the final activation.

## 3. Training Process

-   **Loss Function**: Mean Squared Error (MSE). We want the pixel values of the reconstructed image to be as close as possible to the original.
-   **Optimizer**: Adam. A standard, efficient optimizer for deep learning.
-   **Data Split**: 
    -   Training set (50k images) is used to update weights.
    -   Test set (10k images) is used to monitor validation loss and prevent overfitting.
-   **Callbacks**:
    -   `EarlyStopping`: Stops training if validation loss stops improving.
    -   `ModelCheckpoint`: Saves the best model during training.
    -   `ReduceLROnPlateau`: Reduces learning rate if training plateaus.

## 4. Evaluation and Comparison

### Comparison with PCA
Principal Component Analysis (PCA) is a linear dimensionality reduction technique. We compare our non-linear Autoencoder against PCA to demonstrate the benefits of deep learning for complex data like images.
-   We match the number of PCA components to the Autoencoder's latent dimension.
-   We compare Reconstruction MSE and visual quality.

### Latent Space Classification
To verify if the Autoencoder has learned meaningful features, we use the latent representations $z$ as input to classifiers:
1.  **Neural Network**: A small MLP trained on $z$.
2.  **K-Nearest Neighbors (KNN)**: Non-parametric classifier.
3.  **Nearest Centroid**: Simple distance-based classifier.

High classification accuracy indicates that the encoder has learned to cluster similar images together in the latent space.
