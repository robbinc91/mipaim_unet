# -*- coding: utf-8 -*-

from keras.layers import Input, Conv3D, Add, SpatialDropout3D
from keras.models import Model

from ..configs import Configuration
from .models import ConvBlockFactory


class EncoderFactory(ConvBlockFactory):
    """Create encoder model"""
    pass


class ResidueEncoderFactory(EncoderFactory):
    """Create an encoder that uses a residue block

    input -> stride-2 conv -> norm -> activ -> elem_sum -> norm -> activ
                    |                              |
                    --------------------------------

    """
    def create(self, input_shape, num_out_features, use_dropout=False):
        config = Configuration()
        input = Input(shape=input_shape)
        identity = Conv3D(num_out_features, self.kernel_size, strides=(2, 2, 2),
                          padding='same', use_bias=False,
                          kernel_initializer=self.kernel_initialization)(input)
        residue = self.Normalization()(identity)
        residue = self.Activation()(residue)
        residue = Conv3D(num_out_features, self.kernel_size,
                         padding='same', use_bias=False,
                         kernel_initializer=self.kernel_initialization)(residue)
        output = Add()([identity, residue])
        output = self.Normalization()(output)
        output = self.Activation()(output)
        if use_dropout:
            output = SpatialDropout3D(rate=self.dropout_rate,
                                      data_format=config.data_format)(output)
        model = Model(inputs=input, outputs=output)
        return model
