from typing import List
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_contrib.layers import InstanceNormalization

def ConvBlock(ch_out, input_tensor):
    x = layers.Conv3D(
        ch_out, kernel_size=3, strides=1, padding='same', use_bias=False,
        data_format='channels_first'
    )(input_tensor)
    x = InstanceNormalization( dtype='float32')(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(
        ch_out, kernel_size=3, strides=1, padding='same', use_bias=False,
        data_format='channels_first'
    )(x)
    x = InstanceNormalization( dtype='float32')(x)
    x = layers.ReLU()(x)
    return x

def EncodeBlock(ch_out, input_tensor):
    skip = ConvBlock(ch_out, input_tensor)
    h = layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2),
        data_format='channels_first'
    )(skip)
    return h, skip

def DecodeBlock(ch_out, input_tensor, skip_tensor):
    h = layers.Conv3DTranspose(
        ch_out, kernel_size=2, strides=2, padding='valid', use_bias=False,
        data_format='channels_first'
    )(input_tensor)
    
    h = InstanceNormalization( dtype='float32')(h)
    h = layers.ReLU()(h)
    
    h = tf.concat([h, skip_tensor], axis=1)
    h = ConvBlock(ch_out, h)
    return h

def OpenMapT1UNet(ch_out, input_shape):
    inputs = layers.Input(shape=input_shape, dtype='float32')
    
    # Encoder path
    x = layers.Conv3D(
        32, kernel_size=1, strides=1, padding='same', use_bias=False,
        data_format='channels_first'
    )(inputs)

    x = InstanceNormalization( dtype='float32')(x)
    x = layers.ReLU()(x)
    
    x, skip1 = EncodeBlock(32, x)
    x, skip2 = EncodeBlock(64, x)
    x, skip3 = EncodeBlock(128, x)
    x, skip4 = EncodeBlock(256, x)
    
    # Bottleneck
    x = ConvBlock(512, x)
    
    # Decoder path
    x = DecodeBlock(256, x, skip4)
    x = DecodeBlock(128, x, skip3)
    x = DecodeBlock(64, x, skip2)
    x = DecodeBlock(32, x, skip1)
    
    # Final convolution
    outputs = layers.Conv3D(
        ch_out, kernel_size=1, strides=1, padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
        data_format='channels_first', activation='relu'
    )(x)

    x = InstanceNormalization( dtype='float32')(x)

    outputs = layers.Softmax(axis=1, dtype='float32')(outputs)
    
    return Model(inputs=inputs, outputs=outputs)