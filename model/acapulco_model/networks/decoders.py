# -*- coding: utf-8 -*-

from keras.layers import Input, Conv3D, Concatenate, SpatialDropout3D
from keras.layers import Conv3DTranspose, UpSampling3D
from keras.models import Model

from ..configs import Configuration
from .models import ConvBlockFactory


class DecoderFactory(ConvBlockFactory):
    """Create decoder models"""
    pass


class ShortcutDecoderFactory(DecoderFactory):
    def create(self, input_shape, shortcut_shape, num_out_features):
        """Abstract method to create a shortcut decoder model

        Args:
            input_shape ((3,) tuple of int): The shape of the input
            shortcut_shape ((3,) tuple of int): The shape of the shortchut input
            num_out_features (int): The number of features of the output

        Returns:
            model (keras.Model): The created decoder

        """
        raise NotImplementedError


class UNetDecoderFactory(ShortcutDecoderFactory):
    """Create a UNet decoder
    
        input -> conv 1 -> norm -> acitv -> up -> conv -> norm -> activ -> cat
                                                                            |  
        shortchut -----------------------------------------------------------

        -> conv -> norm -> activ

    """
    def create(self, input_shape, shortcut_shape, num_out_features):
        config = Configuration()
        input = Input(shape=input_shape)
        shortcut = Input(shape=shortcut_shape)
        output = Conv3D(num_out_features, 1, padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(input)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        output = UpSampling3D(size=(2, 2, 2))(output)
        output = Conv3D(num_out_features, self.kernel_size,
                        padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(output)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        output = Concatenate(axis=config.channel_axis)([shortcut, output])
        output = SpatialDropout3D(rate=self.dropout_rate,
                                  data_format=config.data_format)(output)
        output = Conv3D(num_out_features, self.kernel_size,
                        padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(output)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        model = Model(inputs=[input, shortcut], outputs=output)
        return model


class ConvTransposeDecoderFactory(ShortcutDecoderFactory):

    def create(self, input_shape, shortcut_shape, num_out_features):
        config = Configuration()
        input = Input(shape=input_shape)
        shortcut = Input(shape=shortcut_shape)
        output = Conv3D(num_out_features, 1, padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(input)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        output = Conv3DTranspose(num_out_features, self.kernel_size, strides=2,
                                 padding='same', use_bias=False,
                                 kernel_initializer=self.kernel_initialization)(output)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        output = Concatenate(axis=config.channel_axis)([shortcut, output])
        output = SpatialDropout3D(rate=self.dropout_rate,
                                  data_format=config.data_format)(output)
        output = Conv3D(num_out_features, self.kernel_size,
                        padding='same', use_bias=False,
                        kernel_initializer=self.kernel_initialization)(output)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        model = Model(inputs=[input, shortcut], outputs=output)
        return model
