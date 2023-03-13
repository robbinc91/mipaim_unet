import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling3D, UpSampling3D, Concatenate, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Conv3D, AveragePooling3D


def inception_block(inputs, filters):
    conv1 = Conv3D(filters, kernel_size=1, activation="relu")(inputs)
    conv3 = Conv3D(filters, kernel_size=3,
                   activation="relu", padding="same")(inputs)
    conv5 = Conv3D(filters, kernel_size=5,
                   activation="relu", padding="same")(inputs)
    avgpool = AveragePooling3D(pool_size=(
        3, 3, 3), strides=(1, 1, 1), padding="same")(inputs)
    concat = Concatenate()([conv1, conv3, conv5, avgpool])
    return concat


def upsample_block(inputs, skip_features, filters, kernel_size=3):
    x = UpSampling3D(size=(2, 2, 2))(inputs)
    x = Conv3D(filters, kernel_size, activation="relu", padding="same")(x)
    x = LayerNormalization()(x)
    x = Concatenate()([x, skip_features])
    x = inception_block(x, filters)
    return x


def unet_transformer_inception(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    rescaled = Rescaling(scale=1./255)(inputs)

    # UNet encoding path with Inception blocks
    inc1 = inception_block(rescaled, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(inc1)
    inc2 = inception_block(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(inc2)
    inc3 = inception_block(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(inc3)
    inc4 = inception_block(pool3, 256)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(inc4)

    # Transformer
    effnet = EfficientNetB0(include_top=False, input_tensor=pool4)
    transformer = Model(inputs=effnet.input, outputs=effnet.get_layer(
        "block6a_expand_activation").output)
    transformer.trainable = False
    transformer_output = transformer(pool4)

    # UNet decoding path with skip connections and transformer output
    up5 = upsample_block(transformer_output, inc4, 256)
    up6 = upsample_block(up5, inc3, 128)
    up7 = upsample_block(up6, inc2, 64)
    up8 = upsample_block(up7, inc1, 32)

    # Output layer with sigmoid activation for multilabel segmentation
    output = Conv3D(num_classes, kernel_size=1, activation="sigmoid")(up8)

    model = Model(inputs=inputs, outputs=output)
    return model
