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
    output_folder = 'weights/unet_3d_inception/20210809/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 400 epochs per label
    EPOCHS = 1000

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=30,
                                                           restore_best_weights=True,
                                                           baseline=0.09)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/all'
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(step_decay)
    _label = 'parcellationx'

    __labels = [i for i in range(1,29)]

    __output_folder = '{0}{1}/'.format(output_folder, _label)
    if not os.path.exists(__output_folder):
        os.mkdir(__output_folder)
    model_ = inception_unet_semantic_segmentation(shape=REDUCED_MNI_SHAPE_CERSEGSYS_PARCELLATION,
                                                  only_3x3_filters=ONLY_3X3_FILTERS,
                                                  dropout=0.3,
                                                  filters_dim=[32, 32, 32, 64, 64],
                                                  num_labels=28)
    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    partition, outputs = create_cersegsys_partitions(label=_label, use_augmentation=True)
    train_generator = DataGenerator(partition['train'],
                                    outputs,
                                    batch_size=1,
                                    root=CERSEGSYS_2_ROOT,
                                    shuffle=True,
                                    histogram_equalization=True,
                                    in_folder='preprocessed_parcellation_final_2',
                                    binary=True,
                                    labels=__labels)
    val_generator = DataGenerator(partition['validation'],
                                  outputs,
                                  batch_size=1,
                                  root=CERSEGSYS_2_ROOT,
                                  shuffle=True,
                                  histogram_equalization=True,
                                  in_folder='preprocessed_parcellation_final_2',
                                  binary=True,
                                  labels=__labels)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        # Use Dice and jaccard Scores
        __output_folder + 'modelbin.epoch={epoch:03d}.val_dice={val_dice_coefficient:.5f}.h5',
        monitor='val_dice_coefficient',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        period=2
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
    model_.save(__output_folder+'model-{0}.h5'.format(_label))

