import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_autoencoder(latent_dim=128):
    """
    Builds a convolutional autoencoder for 32x32x3 RGB images.
    """
    # ENCODER
    encoder_input = layers.Input(shape=(32, 32, 3), name="encoder_input")
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space representation
    shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
    x = layers.Flatten()(x)
    latent_repr = layers.Dense(latent_dim, name="latent_space")(x)
    
    encoder = models.Model(encoder_input, latent_repr, name="encoder")
    
    # DECODER
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(x)
    
    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    
    # AUTOENCODER
    autoencoder_input = layers.Input(shape=(32, 32, 3), name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    
    return autoencoder, encoder, decoder
