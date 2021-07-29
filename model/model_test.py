from model import parcellation_inception_unet, parcellation_inception_unet_reduced
from common import *

if __name__ == '__main__':
    _model = parcellation_inception_unet_reduced(only_3x3_filters=True, final_droput=0.2)
    _model.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    _model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(_model, to_file='full_parcellation_model_reduced.png', show_shapes=True)
