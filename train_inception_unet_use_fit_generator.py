import tensorflow as tf
from model import inception_unet
from utils import *
import pickle
import keras
from common import *



if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    #model_ = inception_unet()
    #model_ = inception_unet(shape=MNI_SHAPE, only_3x3_filters=ONLY_3X3_FILTERS, dropout=0.01)
    model_ = inception_unet(shape=REDUCED_MNI_SHAPE, only_3x3_filters=ONLY_3X3_FILTERS)
    #model_ = inception_unet(shape=(1, 192, 224))


    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    partition, outputs = create_hammers_partitions()
    train_generator = DataGenerator(partition['train'], outputs, batch_size=1, root=HAMMERS_ROOT, shuffle=True)
    val_generator = DataGenerator(partition['validation'], outputs, batch_size=1, root=HAMMERS_ROOT, shuffle=True)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=10,
                                                           restore_best_weights=True,
                                                           baseline=0.07)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'weights/unet_3d_inception/all/model.epoch={epoch:03d}.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
        monitor='val_dice_coefficient',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        period=1
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs\\all'
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(step_decay)

    callbacks = [
        model_checkpoint_callback,
        #early_stop_callback,
        tensorboard_callback,
        learning_rate_callback
    ]

    print('start fitting')

    history = model_.fit_generator(generator=train_generator, validation_data=val_generator, epochs=EPOCHS, use_multiprocessing=True,
                         callbacks=callbacks)

    with open('history/unet_3d_inception_trainHistoryDict' + str(label) + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



