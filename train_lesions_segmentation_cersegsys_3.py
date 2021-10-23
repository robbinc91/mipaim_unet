import tensorflow as tf
from model import inception_unet
from utils import *
import pickle
import keras
from common import *
import json
import os



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    LABELS = json.load(open('labels.json'))['labels']
    output_folder = 'weights/unet_3d_inception/20211022/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    _label = 'lesions'

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)
    model_ = inception_unet(shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
                                                  only_3x3_filters=ONLY_3X3_FILTERS,
                                                  dropout=0.3,
                                                  filters_dim=[16, 16, 32, 64, 64],
                                                  instance_normalization=True)
    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(label=_label, use_augmentation=True)
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_3_ROOT,
                                    shuffle=True,
                                    histogram_equalization=False,
                                    in_folder='10_final',
                                    is_segmentation=True,
                                    binary=True)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_3_ROOT,
                                  shuffle=True,
                                  histogram_equalization=False,
                                  in_folder='10_final',
                                  is_segmentation=True,
                                  binary=True)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        # Use Dice and jaccard Scores
        __output_folder + 'best.h5',
        monitor='val_soft_dice_score',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        period=4
    )

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        # tensorboard_callback,
        # learning_rate_callback
    ]

    print('start fitting on {0}'.format(_label))

    history = model_.fit_generator(generator=train_generator,
                                   validation_data=val_generator,
                                   epochs=EPOCHS,
                                   use_multiprocessing=False,
                                   callbacks=callbacks)
    #model_.save(__output_folder+'model-{0}.h5'.format(_label))


