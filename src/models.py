# models.py - Ορισμοί αρχιτεκτονικών νευρωνικών δικτύων
"""
Περιέχει τις αρχιτεκτονικές Autoencoder για CIFAR-10:
1. build_autoencoder() - Βασικός CNN autoencoder
2. build_resnet_mae() - ResNet-style Masked Autoencoder (κύρια υλοποίηση)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from config import LATENT_DIM


def build_autoencoder(latent_dim=LATENT_DIM):
    """
    Βασικός Συνελικτικός Αυτοκωδικοποιητής για εικόνες 32x32x3.
    
    Αρχιτεκτονική:
        Encoder: Conv(32) -> Conv(64) -> Conv(128) -> Dense(latent_dim)
        Decoder: Dense -> DeConv(128) -> DeConv(64) -> DeConv(32) -> Conv(3)
    
    Args:
        latent_dim: Διάσταση του bottleneck (default από config)
    
    Returns:
        autoencoder, encoder, decoder: Τα τρία μοντέλα
    """
    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=(32, 32, 3), name="encoder_input")
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 16x16
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 8x8
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 4x4
    
    shape_before_flattening = (4, 4, 128)
    x = layers.Flatten()(x)
    latent_repr = layers.Dense(latent_dim, name="latent_space")(x)
    
    encoder = models.Model(encoder_input, latent_repr, name="encoder")
    
    # ==================== DECODER ====================
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # 8x8
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # 16x16
    
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # 32x32
    
    decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(x)
    
    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    
    # ==================== AUTOENCODER ====================
    autoencoder_input = layers.Input(shape=(32, 32, 3), name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    
    return autoencoder, encoder, decoder


def build_resnet_mae(latent_dim=LATENT_DIM):
    """
    ResNet-style Masked Autoencoder για CIFAR-10.
    
    Αρχιτεκτονική:
        Encoder:
            - Initial Conv(64)
            - ResBlock(64) -> ResBlock(128, stride=2) -> ResBlock(256, stride=2) -> ResBlock(512, stride=2)
            - Flatten -> Dense(latent_dim)
        
        Decoder:
            - Dense -> Reshape(4, 4, 512)
            - DeConv(256) -> DeConv(128) -> DeConv(64)
            - Conv(3, tanh)
    
    Args:
        latent_dim: Διάσταση του bottleneck
    
    Returns:
        autoencoder, encoder, decoder: Τα τρία μοντέλα
    """
    def res_block(x, filters, stride=1):
        """Residual Block με skip connection."""
        shortcut = x
        
        # Κύρια διαδρομή
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut projection αν αλλάζουν οι διαστάσεις
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=(32, 32, 3), name="encoder_input")
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks με progressive downsampling
    x = res_block(x, 64)                  # 32x32
    x = res_block(x, 128, stride=2)       # 16x16
    x = res_block(x, 128)
    x = res_block(x, 256, stride=2)       # 8x8
    x = res_block(x, 256)
    x = res_block(x, 512, stride=2)       # 4x4
    x = res_block(x, 512)
    
    # Bottleneck
    shape_before_flattening = (4, 4, 512)  # 8192 features
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent_space")(x)
    
    encoder = models.Model(encoder_input, latent, name="encoder")
    
    # ==================== DECODER ====================
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    
    # Upsampling με Conv2DTranspose
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)  # 8x8
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)  # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)  # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Refinement layer
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Output layer με tanh για [-1, 1]
    decoder_output = layers.Conv2D(3, (3, 3), activation='tanh', padding='same', name="decoder_output")(x)
    
    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    
    # ==================== AUTOENCODER ====================
    autoencoder_input = layers.Input(shape=(32, 32, 3), name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = models.Model(autoencoder_input, decoded, name="resnet_mae")
    
    # Print model summary
    print(f"\n=== ResNet-MAE Architecture (latent_dim={latent_dim}) ===")
    print(f"Encoder: {encoder_input.shape} -> {latent.shape}")
    print(f"Decoder: {decoder_input.shape} -> {decoder_output.shape}")
    print(f"Total parameters: {autoencoder.count_params():,}")
    
    return autoencoder, encoder, decoder


# Alias για backward compatibility
build_conv_autoencoder = build_autoencoder
