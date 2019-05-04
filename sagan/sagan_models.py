import numpy as np
import tensorflow as tf
from spectral import SpectralConv2D, SpectralConv2DTranspose
from attention import SelfAttnModel


def create_generator(image_size=64, z_dim=100, filters=64, kernel_size=4):
        
    input_layers = tf.keras.layers.Input((z_dim,))
    x = tf.keras.layers.Reshape((1, 1, z_dim))(input_layers)

    filters = 256
    for i in range(3):
        print(filters, kernel_size)
        x = SpectralConv2DTranspose(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=4, 
                                    padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        filters //= 2

    x, attn1 = SelfAttnModel(64)(x)

    x = SpectralConv2DTranspose(filters=32, 
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = SpectralConv2DTranspose(filters=16, 
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x, attn2 = SelfAttnModel(32)(x)
    x = SpectralConv2DTranspose(filters=3, 
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                name='last_deconv')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    return tf.keras.models.Model(input_layers, [x, attn1, attn2])


def create_discriminator(image_size=64, filters=64, kernel_size=4):
    input_layers = tf.keras.layers.Input((image_size, image_size, 3))

    x = input_layers
    filters = 8
    for i in range(3):
        x = SpectralConv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=2,
                           padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        filters = filters * 2

    # 512 -> 64
        
    x, attn1 = SelfAttnModel(32)(x)

    x = SpectralConv2D(filters=64,
                        kernel_size=kernel_size,
                        strides=4,
                        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = SpectralConv2D(filters=128,
                        kernel_size=kernel_size,
                        strides=4,
                        padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    
    x, attn2 = SelfAttnModel(128)(x)

    x = SpectralConv2D(filters=1, kernel_size=4, strides=4, padding='same', name='last_conv')(x)
    x = tf.keras.layers.Flatten()(x)
    
    return tf.keras.models.Model(input_layers, [x, attn1, attn2])
