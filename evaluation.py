import keras
from common import *
from utils.losses import *
from utils.preprocess import *
from keras.models import load_model


def evaluate():
    t1, outputs = hammers_2017_data_evaluation_reduced(HAMMERS_ROOT)
    model = load_model('weights/unet_3d_inception/labelinc/best2.h5',
                       custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})

    X = []
    y = []

    for t1_, seg_ in zip(t1, outputs):
        # T1 = histeq(to_uint8(get_data_with_skull_scraping(t1_)))
        T1 = to_uint8(get_data(t1_))
        X.append(T1[None, ...])

        # y.append(np.array(get_data(seg_) == label).astype(np.uint8)[None, ...])
        y_ = to_uint8(get_data(seg_))
        y.append(y_[None, ...])

    X = np.array(X)
    y = np.array(y)

    eval = model.evaluate(x=X, y=y)
    print(eval)



if __name__ == '__main__':
    evaluate()
