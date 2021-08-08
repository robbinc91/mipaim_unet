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
    model_ = inception_unet(shape=REDUCED_MNI_SHAPE_MINE , only_3x3_filters=ONLY_3X3_FILTERS, dropout=0.2, filters_dim=[8, 16, 32, 64, 128])
    #model_ = inception_unet(shape=(1, 192, 224))


    model_.compile(optimizer='adam',
                   loss=dice_loss,
                   metrics=[dice_coefficient])
    model_.summary()

    #partition, outputs = create_hammers_partitions()
    # use new MRIs
    partition, outputs = create_hammers_partitions_new(use_augmentation=True)
    train_generator = DataGenerator(partition['train'], outputs, batch_size=1, root=MY_ROOT, shuffle=True, histogram_equalization=True, in_folder='reduced') # HAMERS_ROOT
    val_generator = DataGenerator(partition['validation'], outputs, batch_size=1, root=MY_ROOT, shuffle=True, histogram_equalization=True, in_folder='reduced') # HAMMERS_ROOT

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=30,
                                                           restore_best_weights=True,
                                                           baseline=0.09)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'weights/unet_3d_inception/20210804-segmentation/model-big.epoch={epoch:03d}.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
        monitor='val_dice_coefficient',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='max',
        period=4
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/all'
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(step_decay)

    callbacks = [
        model_checkpoint_callback,
        #early_stop_callback,
        #tensorboard_callback,
        #learning_rate_callback
    ]

    print('start fitting')
    EPOCHS = 245
    
    history = model_.fit_generator(generator=train_generator, validation_data=val_generator, epochs=EPOCHS, use_multiprocessing=True,
                         callbacks=callbacks)

    with open('history/unet_3d_inception_trainHistoryDict' + str(label) + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



