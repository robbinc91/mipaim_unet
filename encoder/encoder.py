from keras.layers import Input

from encoder.t1_encoder import t1_encoder
from encoder.flair_encoder import flair_encoder
from encoder.ir_encoder import ir_encoder


def encode(t1, FLAIR=None, IR=None, IMAGE_ORDERING='channels_first', shape=(1, 240, 240, 28)):

    """
    :param t1: any | none
    :param FLAIR: any | None
    :param IR: any | None
    :param IMAGE_ORDERING: string
    :param shape: 4x tuple
    :return: dict, dict, dict, dict
    """


    img_input_t1 = Input(shape=shape, name='T1') if t1 is not None else None
    maxpool_t1, conv_21_t1, conv_32_t1 = t1_encoder(img_input_t1, IMAGE_ORDERING) if t1 is not None else (None, None, None)

    img_input_FLAIR = Input(shape=shape, name="FLAIR") if FLAIR is not None else None
    maxpool_FLAIR, conv_21_FLAIR, conv_32_FLAIR = flair_encoder(img_input_FLAIR, IMAGE_ORDERING) if FLAIR is not None else (None, None, None)

    img_input_IR = Input(shape=shape, name="IR") if IR is not None else None
    maxpool_IR, conv_21_IR, conv_32_IR = ir_encoder(img_input_IR, IMAGE_ORDERING) if IR is not None else (None, None, None)



    return {'t1_input': img_input_t1, 'flair_input': img_input_FLAIR, 'ir_input': img_input_IR},\
           {'t1_output': maxpool_t1, 'flair_output': maxpool_FLAIR, 'ir_output': maxpool_IR}, \
           {'conv_21_t1': conv_21_t1, 'conv_21_FLAIR': conv_21_FLAIR, 'conv_21_IR': conv_21_IR}, \
           {'conv_32_t1': conv_32_t1, 'conv_32_FLAIR': conv_32_FLAIR, 'conv_32_IR': conv_32_IR}
