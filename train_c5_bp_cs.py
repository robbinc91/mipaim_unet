from email import generator
import tensorflow as tf
from model import mipaim_unet
from utils import *
import keras
from common import *
import json
import os

#from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# input()

if __name__ == '__main__':

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.enable_eager_execution()

    LABELS = json.load(open('labels_c5_bp_cs.json'))['labels']
    output_folder = 'weights/mipaim_unet/20221022/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    _label = 'bp_cs'

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.makedirs(__output_folder)
    model_ = mipaim_unet(
        shape=REDUCED_MNI_SHAPE_MINE,
        only_3x3_filters=ONLY_3X3_FILTERS,
        dropout=0.3,
        filters_dim=[2, 2, 2, 4, 4],
        instance_normalization=True,
        num_labels=4)
    model_.compile(optimizer='adam',
                   loss=soft_dice_loss,
                   metrics=[soft_dice_score])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(
        label=_label, use_augmentation=True, second_lbl='-clahe')
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_5_ROOT,
                                    shuffle=True,
                                    histogram_equalization=False,
                                    in_folder='08_augmentations',
                                    is_segmentation=True,
                                    binary=False)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_5_ROOT,
                                  shuffle=True,
                                  histogram_equalization=False,
                                  in_folder='08_augmentations',
                                  is_segmentation=True,
                                  binary=False)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        __output_folder +
        'model.epoch={epoch:03d}.val_dice_score={soft_dice_score:.5f}.h5',
        monitor='soft_dice_score',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        #save_freq=20
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/bp_cs'
    )

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        tensorboard_callback,
        learning_rate_callback
    ]

    print('start fitting on {0}'.format(_label))

    model_.fit(x=train_generator,
               y=None,
               batch_size=1,
               validation_data=val_generator,
               epochs=EPOCHS,
               use_multiprocessing=False,
               callbacks=callbacks)
    # history = model_.fit_generator(generator=train_generator,
    #                               validation_data=val_generator,
    #                               epochs=EPOCHS,
    #                               use_multiprocessing=False,
    #                               callbacks=callbacks)
    model_.save(__output_folder+'model-{0}.h5'.format(_label))
