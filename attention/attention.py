from keras.activations import sigmoid
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, \
    GlobalAveragePooling3D, \
    GlobalMaxPooling2D, \
    GlobalMaxPooling3D, \
    Reshape, \
    Dense, \
    multiply, \
    Permute, \
    Concatenate, \
    Conv2D, \
    Conv3D, \
    Add, \
    Activation, \
    Lambda


# TODO: Convert code to work with 2 and 3D

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)

    print(cbam_feature.shape)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    GlobalAveragePooling = GlobalAveragePooling3D if len(input_feature.shape) == 5 else GlobalAveragePooling2D
    GlobalMaxPooling = GlobalMaxPooling3D if len(input_feature.shape) == 5 else GlobalMaxPooling2D


    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    init_tuple = (1, 1, 1, channel) if len(input_feature.shape) == 5 else (1, 1, channel)

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling()(input_feature)
    avg_pool = Reshape(init_tuple)(avg_pool)
    assert avg_pool._keras_shape[1:] == init_tuple
    avg_pool = shared_layer_one(avg_pool)
    #assert avg_pool._keras_shape[1:] == (1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == init_tuple

    max_pool = GlobalMaxPooling()(input_feature)
    max_pool = Reshape(init_tuple)(max_pool)
    assert max_pool._keras_shape[1:] == init_tuple
    max_pool = shared_layer_one(max_pool)
    #assert max_pool._keras_shape[1:] == (1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == init_tuple

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((4, 1, 2, 3))(cbam_feature)

    print(input_feature.shape)
    print(cbam_feature.shape)
    print()
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):

    Conv = Conv3D if len(input_feature.shape) == 5 else Conv2D

    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    
    concat = Permute((4, 2, 3, 1))(concat)
    cbam_feature = Conv(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    cbam_feature = Permute((4, 2, 3, 1))(cbam_feature)
    assert cbam_feature._keras_shape[-1] == 1

        

    print(cbam_feature.shape)
    return multiply([input_feature, cbam_feature])
