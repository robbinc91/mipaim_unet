from model import unet, model_thresholding
from utils import *
import pickle
import keras
from common import *

if __name__ == '__main__':

    #model_ = unet(t1=True, FLAIR=None, IR=None)

    #model_.compile('adam', dice_loss, [dice_coefficient])
    #model_.summary()

    #exit(0)

    #from keras.utils.vis_utils import plot_model

    #plot_model(model_, to_file='3d_unet_model_plot_t1_flair_ir.png', show_shapes=True)


    T1path, FLAIRpath, IRpath, segpath = data_train(root=ROOT)  # TRAIN IMAGES
    T1_val, FLAIR_val, IR_val, segm_val = data_val(root=ROOT)  # TEST IMAGES
    FLAIRpath = None
    IRpath = None

    if label in [1, 3, 5]:
        print("TRAINING ON THRESHOLDING MODEL...")
        print("LOADING DATA...")
        X = []
        y = []
        if label == 5:
            for T1_, seg_ in zip(T1path, segpath):
                T1 = get_data_with_skull_scraping(T1_)
                y.append(np.array(get_data(seg_) == 5).astype(np.uint8)[None, ...])
                X.append(np.array((T1 >= 10) & (T1 < 110)).astype(np.uint8)[None, ...])  # <-Works better
            X = np.array(X)
            y = np.array(y)
            T1 = get_data_with_skull_scraping(T1_val)
            X_val = np.array((T1 >= 10) & (T1 < 110)).astype(np.uint8)[None, None, ...]
            y_val = np.array(get_data(segm_val) == 5).astype(np.uint8)[None, ...]
        elif label == 3:
            for T1_, seg_ in zip(T1path, segpath):
                T1 = get_data_with_skull_scraping(T1_)
                y.append(np.array(get_data(seg_) == 3).astype(np.uint8)[None, ...])
                X.append(np.array(T1 >= 150).astype(np.uint8)[None, ...])
            X = np.array(X)
            y = np.array(y)
            T1 = get_data_with_skull_scraping(T1_val)
            X_val = np.array(T1 >= 150).astype(np.uint8)[None, None, ...]
            y_val = np.array(get_data(segm_val) == 3).astype(np.uint8)[None, ...]
        else:
            for T1_, seg_ in zip(T1path, segpath):
                T1 = get_data_with_skull_scraping(T1_)
                y.append(np.array(get_data(seg_) == 1).astype(np.uint8)[None, ...])
                X.append(np.array((T1 >= 80) & (T1 < 160)).astype(np.uint8)[None, ...])
            X = np.array(X)
            y = np.array(y)
            T1 = get_data_with_skull_scraping(T1_val)
            X_val = np.array((T1 >= 80) & (T1 < 160)).astype(np.uint8)[None, None, ...]
            y_val = np.array(get_data(segm_val) == 1).astype(np.uint8)[None, ...]

        print("STARTING TRAINING...")
        model_ = model_thresholding()
        model_.compile('adam', dice_loss, [dice_coefficient])
        model_.summary()
        history = model_.fit(X, y, validation_data=(X_val, y_val), epochs=EPOCHS,
                             callbacks=[keras.callbacks.ModelCheckpoint(
                                 'weights/unet_3d_t1/label' + str(label) + '/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                                 monitor='val_dice_coefficient',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='max',
                                 period=1)])

    else:
        print("LOADING DATA...")
        X_T1 = []
        X_FLAIR = []
        X_IR = []
        y = []

        X_FLAIR_val = None
        X_IR_val = None

        TRAIN_ARRAY = []
        EVAL_ARRAY = []

        for T1_, seg_ in zip(T1path, segpath):
            T1 = histeq(to_uint8(get_data_with_skull_scraping(T1_)))
            X_T1.append(T1[None, ...])

            y.append(np.array(get_data(seg_) == label).astype(np.uint8)[None, ...])

        X_T1 = np.array(X_T1)
        y = np.array(y)
        X_T1_val = histeq(to_uint8(get_data_with_skull_scraping(T1_val)))[None, None, ...]

        TRAIN_ARRAY.append(X_T1)
        EVAL_ARRAY.append(X_T1_val)

        # TODO: Test this
        y_val = np.array(get_data(segm_val) == label).astype(np.uint8)[None, ...]
        # y_val = np.array(get_data(segm_val) == CLASS).astype(np.uint8)[None, ...]

        if FLAIRpath is not None and IRpath is not None:
            for FLAIR_, IR_ in zip(FLAIRpath, IRpath):
                IR = IR_to_uint8(get_data(IR_))
                FLAIR = to_uint8(get_data(FLAIR_))

                X_IR.append(IR[None, ...])
                X_FLAIR.append(FLAIR[None, ...])


            X_FLAIR = np.array(X_FLAIR)
            X_IR = np.array(X_IR)


            X_FLAIR_val = to_uint8(get_data(FLAIR_val))[None, None, ...]
            X_IR_val = IR_to_uint8(get_data(IR_val))[None, None, ...]

            TRAIN_ARRAY.append(X_FLAIR)
            TRAIN_ARRAY.append(X_IR)

            EVAL_ARRAY.append(X_FLAIR_val)
            EVAL_ARRAY.append(X_IR_val)
        else:
            TRAIN_ARRAY = TRAIN_ARRAY[0]
            EVAL_ARRAY = EVAL_ARRAY[0]

        HAS_FLAIR = True if FLAIRpath is not None else None
        HAS_IR = True if IRpath is not None else None
        print("STARTING TRAINING...")
        model_ = vnet(t1=True, FLAIR=HAS_FLAIR, IR=HAS_IR)

        model_.compile('adam', dice_loss, [dice_coefficient])
        model_.summary()



        history = model_.fit(TRAIN_ARRAY, y=y, validation_data=(EVAL_ARRAY, y_val),
                             epochs=EPOCHS, callbacks=[keras.callbacks.ModelCheckpoint(
                'weights/unet_3d_t1/label' + str(label) + '/Model.val_dice_coefficient={val_dice_coefficient:.5f}.h5',
                monitor='val_dice_coefficient',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                period=1
            )])

    with open('history/unet_3d_t1_trainHistoryDict' + str(label) + '.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

