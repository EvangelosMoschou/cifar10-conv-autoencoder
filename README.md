# Convolutional Autoencoder for CIFAR-10

This project implements a Convolutional Autoencoder for the CIFAR-10 dataset using TensorFlow/Keras. It serves as the 3rd Mandatory Assignment for the Neural Networks & Deep Learning course.

## Project Overview

The main objectives of this project are:
1.  **Image Reconstruction**: Train an autoencoder to reconstruct CIFAR-10 images.
2.  **Dimensionality Reduction**: Compare the autoencoder's compression capabilities against Principal Component Analysis (PCA).
3.  **Feature Extraction**: Use the learned latent space for classification tasks (Neural Network, KNN, Nearest Centroid).

## Structure or the Project

```
Project 3/
├── Cifar10/             # CIFAR-10 data handling
├── assignment/          # Assignment specifications
├── data/                # Data storage
├── src/                 # Source code
│   ├── models.py        # Model definitions (Autoencoder, Classifier)
│   ├── data_utils.py    # Data loading and preprocessing
│   ├── train_autoencoder.py # Training script
│   └── evaluate_autoencoder.py # Evaluation script
├── results/             # Generated results (checkpoints, logs, plots)
├── Report/              # LaTeX report files
└── README.md            # This file
```

## Setup

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn pandas tqdm seaborn
    ```

## Usage

### Training the Autoencoder

Run the training script to train the autoencoder model:

```bash
python src/train_autoencoder.py
```

This will save the best model to `results/best_autoencoder.h5` and log training metrics.

### Evaluation and Comparison

Run the evaluation script to generate reconstruction images, compare with PCA, and run classification benchmarks:

```bash
python src/evaluate_autoencoder.py
```

Results will be saved in the `results/` directory, including:
- `reconstructions.png`: Visual comparison of original vs. reconstructed images.
- `pca_comparison.png`: Comparison between Autoencoder and PCA reconstructions.
- `metrics.csv`: Quantitative results.

## Author
[Your Name/Student ID]
