# -*- coding: utf-8 -*-

from keras.layers import Input, Conv3D
from keras.models import Model

from .models import ConvBlockFactory


class InputFactory(ConvBlockFactory):

    def create(self, input_shape, num_out_features):
        """Create a input block

        conv -> norm -> activ

        """
        input = Input(shape=input_shape)
        output = Conv3D(num_out_features, self.kernel_size, padding='same',
                        use_bias=False,
                        kernel_initializer=self.kernel_initialization)(input)
        output = self.Normalization()(output)
        output = self.Activation()(output)
        model = Model(inputs=input, outputs=output)
        return model
