import tensorflow as tf
from model import inception_unet_semantic_segmentation
from utils import *
import pickle
import keras
from common import *
import json
import os



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    LABELS = json.load(open('labels.json'))['labels']
    output_folder = 'weights/unet_3d_inception/20211028/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 400

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=30,
                                                           restore_best_weights=True,
                                                           baseline=0.09)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/all'
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(step_decay)
    _label = 'parcellation_3lb'

    __labels = [i for i in range(1, 4)]

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)
    model_ = inception_unet_semantic_segmentation(shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
                                                  only_3x3_filters=ONLY_3X3_FILTERS,
                                                  dropout=0.3,
                                                  filters_dim=[8, 8, 16, 32, 32],
                                                  num_labels=len(__labels),
                                                  instance_normalization=True)
    model_.compile(optimizer='adam',
                   loss=soft_dice_loss,
                   metrics=[soft_dice_score])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(label=_label, use_augmentation=False, second_lbl='-histeq')
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_4_ROOT,
                                    shuffle=True,
                                    histogram_equalization=False,
                                    in_folder='10_final',
                                    binary=True,
                                    labels=__labels)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_4_ROOT,
                                  shuffle=True,
                                  histogram_equalization=False,
                                  in_folder='10_final',
                                  binary=True,
                                  labels=__labels)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        # Use Dice and jaccard Scores
        __output_folder + 'binpmodel.epoch={epoch:03d}.val_dice_coefficient={val_soft_dice_score:.5f}.h5',
        save_best_only=False,
        save_weights_only=False,
        period=10
    )

    callbacks = [
        model_checkpoint_callback,
        # early_stop_callback,
        # tensorboard_callback,
        # learning_rate_callback
    ]

    print('start fitting on {0}'.format('full cerebellum parcellation'))

    history = model_.fit_generator(generator=train_generator,
                                   validation_data=val_generator,
                                   epochs=EPOCHS,
                                   use_multiprocessing=False,
                                   callbacks=callbacks)
    model_.save(__output_folder+'binpmodel-{0}.h5'.format(_label))


