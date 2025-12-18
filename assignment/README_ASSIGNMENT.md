# 3η Υποχρεωτική Εργασία - Νευρωνικά Δίκτυα & Βαθιά Μάθηση

## LLM-Readable Assignment Summary

### Course Information
- **Course**: Neural Networks – Deep Learning (Νευρωνικά Δίκτυα – Βαθιά Μάθηση)
- **Assignment**: 3rd Mandatory Assignment (3η Υποχρεωτική Εργασία)
- **Deadline**: January 11, 2026
- **Penalty**: 10% grade reduction per day late (up to 5 days)

---

## Assignment Objective

Implement an **Autoencoder Neural Network** in Python (using TensorFlow/Keras) that will be trained for:
- **Image reconstruction** on the CIFAR-10 dataset
- **Feature extraction** for classification tasks
- **Comparison with PCA** for dimensionality reduction

**Why Autoencoder?**
- Industry standard for unsupervised feature learning
- Widely used in production systems (anomaly detection, compression, generation)
- Excellent for demonstrating deep learning concepts
- Aligns with existing project infrastructure (CIFAR-10)
- TensorFlow/Keras provides excellent built-in CIFAR-10 dataset support

---

## Dataset: CIFAR-10

**Source**: https://www.cs.toronto.edu/~kriz/cifar.html

**Description**:
- 60,000 32×32 color images in 10 classes
- 50,000 training images
- 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Tasks**:
1. **Primary**: Image reconstruction using Autoencoder
2. **Secondary**: Multi-class classification using learned latent features
3. **Comparison**: PCA-based reconstruction vs Autoencoder reconstruction

---

## Data Splitting Requirements

If no test set is provided:
- **Training set**: 60%
- **Test set**: 40%
- **Alternative**: Use cross-validation technique

---

## Feature Extraction (Optional)

For sample separation/classification:
- Use **PCA** to reduce dimensionality
- Retain **≥ 90%** of the information

---

## Required Report Contents

The report must include:

### 1. Algorithm Description
- Detailed explanation of the implemented algorithm

### 2. Classification/Reconstruction Examples
- Examples of correct classifications/reconstructions
- Examples of incorrect classifications/reconstructions

### 3. Performance Metrics
- Success rates during training
- Success rates during testing
- Training time

### 4. Hyperparameter Analysis
- Results with different latent space dimensions (e.g., 32, 64, 128, 256)
- Different encoder/decoder architectures
- Different learning rates and optimizers
- Different batch sizes
- Training epochs and convergence analysis

### 5. Comparative Analysis

**Required Comparisons**:

1. **Autoencoder vs PCA Reconstruction**
   - Reconstruction quality (MSE, SSIM)
   - Visual quality comparison
   - Computational efficiency
   - Latent space dimensionality impact

2. **Classification Performance**
   - Autoencoder latent features vs raw pixels
   - Autoencoder latent features vs PCA features
   - Nearest Neighbor classifier on different feature spaces
   - Nearest Class Centroid classifier on different feature spaces

### 6. Special Evaluations

**For CIFAR-10 Autoencoder**:
- Evaluate reconstruction quality across different object classes
- Analyze which classes are easier/harder to reconstruct
- Visualize latent space using t-SNE or UMAP
- Test classification accuracy using learned features

### 7. Code Documentation
- Commented source code with explanations

---

## Presentation Requirements

After submission, an oral presentation is required with **8-10 PowerPoint slides**:

| Slide # | Content |
|---------|---------|
| 1 | Student information, datasets used, special considerations |
| 2-9 | Results for each assignment: performance, parameter values, training time, etc. |
| 10 | Overall conclusions and comments |

**Presentation slides must be submitted to e-learning the day before the presentation/exam.**

---

## Key Technical Terms (Greek → English)

| Greek | English |
|-------|---------|
| Νευρωνικά Δίκτυα | Neural Networks |
| Βαθιά Μάθηση | Deep Learning |
| Ακτινικές Βασικές Συναρτήσεις | Radial Basis Functions |
| Αυτοκωδικοποιητής | Autoencoder |
| Εκπαίδευση | Training |
| Έλεγχος | Testing |
| Κρυφοί νευρώνες | Hidden neurons |
| Πλησιέστερος γείτονας | Nearest Neighbor |
| Πλησιέστερο κέντρο κλάσης | Nearest Class Centroid |
| Ανακατασκευή | Reconstruction |
| Κατηγοριοποίηση | Classification |
