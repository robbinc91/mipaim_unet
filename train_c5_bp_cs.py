import tensorflow as tf
from model import mipaim_unet
from utils import *
import pickle
import keras
from common import *
import json
import os


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    LABELS = json.load(open('labels_c5_bp_cs.json'))['labels']
    output_folder = 'weights/mipaim_unet/20221022/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    _label = 'parcellation-x'

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)
    model_ = mipaim_unet(# TODO: Correct shape
        shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
        only_3x3_filters=ONLY_3X3_FILTERS,
        dropout=0.3,
        filters_dim=[8, 16, 32, 64, 128],
        instance_normalization=True,
        num_labels=4)
    model_.compile(optimizer='adam',
                   loss=soft_dice_loss,
                   metrics=[soft_dice_score])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(
        label=_label, use_augmentation=False)
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_4_ROOT,
                                    shuffle=True,
                                    histogram_equalization=False,
                                    in_folder='10_final',
                                    is_segmentation=True,
                                    binary=True)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_4_ROOT,
                                  shuffle=True,
                                  histogram_equalization=False,
                                  in_folder='10_final',
                                  is_segmentation=True,
                                  binary=True)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        __output_folder +
        'model.epoch={epoch:03d}.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
        monitor='val_dice_coefficient',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        period=10
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        # tensorboard_callback,
        learning_rate_callback
    ]

    print('start fitting on {0}'.format(_label))

    history = model_.fit_generator(generator=train_generator,
                                   validation_data=val_generator,
                                   epochs=EPOCHS,
                                   use_multiprocessing=False,
                                   callbacks=callbacks)
    model_.save(__output_folder+'model-{0}.h5'.format(_label))
