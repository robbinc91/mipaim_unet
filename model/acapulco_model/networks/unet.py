# -*- coding: utf-8 -*-

from keras.layers import Input
from keras.models import Model


class UNetFactory:
    """Create U-Net model

    The model is based on Kayalibay et al. CNN-based Segmentation of Medical
    Imaging Data.  https://arxiv.org/pdf/1701.03056.pdf (2017).

    input_block -----------------------------> encoder ----> output_block
          |                                       |      |
          --> encoder -----------------> decoder ---------
                 |                          |       |
                 --> encoder ----> decoder ----------
                          ...    ...    ...
                        |             |       |
                        --> decoder -----------

    Attributes:
        input_factory (.inputs.InputFactory): Create the input block
        encoder_factory (.inputs.EncoderFactory): Create encoders
        decoder_factory (.inputs.DecoderFactory): Create decoders
        output_factory (.inputs.OutputFactory): Create the output block

    """
    def __init__(self, input_factory, encoder_factory, decoder_factory,
                 output_factory, max_num_features=512):
        """Initialize

        Args:
            max_num_features (int): The number of features is doubled but
                limited by this number after each decoders
        
        """
        self.input_factory = input_factory
        self.encoder_factory = encoder_factory
        self.decoder_factory = decoder_factory
        self.output_factory = output_factory

        self.max_num_features = max_num_features
        
    def create(self, input_shape, num_input_block_features, num_classes,
               num_encoders, num_aggregates):
        """Create a U-Net

        Args:
            input_shape ((4,) tuple of int): The shape of the multi-channel 3D
                image to process
            num_input_block_features (int): The number of output features of the
                input block
            num_classes (int): The number of classes
            num_encoders (int): The number of encoders. The same with the number
                of decoders
            num_aggregates (int): The number of encoders that output to the
                output block, should be smaller than `num_encoders - 1`. The
                last `num_aggregates` of encoders are selected.

        Returns:
            model (keras.Model): The U-Net model

        """
        input = Input(shape=input_shape)
        output = self.input_factory.create(input_shape,
                                           num_input_block_features)(input)

        # encoders
        shortcuts = [output]
        output_shape = output.shape[1:].as_list() # remove batch dim
        num_out_features = num_input_block_features
        for i in range(num_encoders):
            output = self.encoder_factory.create(output_shape,
                                                 num_out_features)(output)
            output_shape = output.shape[1:].as_list()
            if i < num_encoders - 1:
                shortcuts.insert(0, output) # stack
            num_out_features = self._get_num_features(2 * num_out_features)

        # decoders
        outputs = list()
        output_shapes = list()
        for i, shortcut in enumerate(shortcuts):
            shortcut_shape = shortcut.shape[1:].as_list()
            num_out_features = shortcut_shape[0]
            decoder = self.decoder_factory.create(output_shape, shortcut_shape,
                                                  num_out_features)
            output = decoder([output, shortcut])
            output_shape = output.shape[1:].as_list()
            if i >= len(shortcuts) - num_aggregates:
                outputs.append(output)
                output_shapes.append(output_shape)

        # output
        output = self.output_factory.create(output_shapes, num_classes)(outputs)
        model = Model(inputs=input, outputs=output)

        return model

    def _get_num_features(self, num_features):
        return min(num_features, self.max_num_features)
