import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, UpSampling3D, Concatenate, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0


def conv_block(inputs, filters, kernel_size=3, activation="relu", padding="same"):
    x = Conv3D(filters, kernel_size, activation=activation,
               padding=padding)(inputs)
    x = LayerNormalization()(x)
    x = Conv3D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = LayerNormalization()(x)
    return x


def upsample_block(inputs, skip_features, filters, kernel_size=3, activation="relu", padding="same"):
    x = UpSampling3D(size=(2, 2, 2))(inputs)
    x = Conv3D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = LayerNormalization()(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, filters, kernel_size, activation, padding)
    return x


def unet_transformer(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    rescaled = Rescaling(scale=1./255)(inputs)

    # UNet encoding path
    conv1 = conv_block(rescaled, 64)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Transformer
    effnet = EfficientNetB0(include_top=False, input_tensor=pool4)
    transformer = Model(inputs=effnet.input, outputs=effnet.get_layer(
        "block6a_expand_activation").output)
    transformer.trainable = False
    transformer_output = transformer(pool4)

    # UNet decoding path with skip connections and transformer output
    up5 = upsample_block(transformer_output, conv4, 512)
    up6 = upsample_block(up5, conv3, 256)
    up7 = upsample_block(up6, conv2, 128)
    up8 = upsample_block(up7, conv1, 64)

    # Output layer with sigmoid activation for multilabel segmentation
    output = Conv3D(num_classes, kernel_size=1, activation="sigmoid")(up8)

    model = Model(inputs=inputs, outputs=output)
    return model
