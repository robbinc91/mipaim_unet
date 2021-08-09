from model import parcellation_inception_unet, \
    parcellation_inception_unet_reduced, \
    classification_model, \
    inception_unet_semantic_segmentation

from common import *

if __name__ == '__main__':
    _model = inception_unet_semantic_segmentation(shape=(1, 128, 80, 80), only_3x3_filters=True, dropout=0.2)
    _model.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    _model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(_model, to_file='parcellating_network_model.png', show_shapes=True)
