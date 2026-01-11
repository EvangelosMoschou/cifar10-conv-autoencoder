# config.py - Κεντρικό αρχείο ρυθμίσεων για το project
"""
Συγκεντρώνει όλες τις παγκόσμιες παραμέτρους σε ένα μέρος
για να υπάρχει συνέπεια μεταξύ όλων των scripts.
"""

# ==================== ΔΕΔΟΜΕΝΑ ====================
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']  # Ονόματα κλάσεων CIFAR-10

# ==================== ΜΟΝΤΕΛΟ ====================
LATENT_DIM = 128  # Διάσταση του bottleneck

# ==================== ΜΑΣΚΑΡΙΣΜΑ ====================
MASK_RATIO = 0.5   # Ποσοστό patches που αποκρύπτονται
PATCH_SIZE = 4     # Μέγεθος κάθε patch (4x4 pixels)

# ==================== ΕΚΠΑΙΔΕΥΣΗ ====================
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5

# ==================== ΔΙΑΔΡΟΜΕΣ ====================
RESULTS_DIR = "results"
MODELS_DIR = "results/models"
PLOTS_DIR = "results/plots"
METRICS_DIR = "results/metrics"

# ==================== ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ ====================
# Τα pixels κανονικοποιούνται από [0, 255] -> [-1, 1]
PIXEL_MIN = -1.0
PIXEL_MAX = 1.0
