import tensorflow as tf
import keras.backend as K
import numpy as np
from functools import partial

def categorical_crossentropy_3d(y_true, y_pred):
    """
    Computes categorical cross-entropy loss for a softmax distribution in a 3D array
    with shape (num_samples, num_classes, dim1, dim2,dim3)

    Parameters
    ----------
    y_true : keras.placeholder [batches,num_classes,dim0,dim1,dim3]
        Placeholder for data holding the ground-truth labels encoded in a one-hot representation
    y_pred : keras.placeholder [batches,num_classes,dim0,dim1,dim3]
        Placeholder for data holding the softmax distribution over classes

    Returns
    -------
    scalar
        Categorical cross-entropy loss value
    """
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
    num_total_elements = K.sum(y_true_flatten)
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy


def dice_loss_multilabel(y_true, y_pred):
    return 1-dice_coefficient_multilabel(y_true, y_pred)


def dice_coefficient_multilabel(y_true, y_pred, smooth=.1):
    dice = 0.
    
    for i in range(y_pred.shape[1]):
        dice += dice_coefficient(y_true[:,i,:,:,:], y_pred[:,i,:,:,:])
    
    dice /= y_pred.shape[1]

    return dice



def dice_coefficient(y_true, y_pred, smooth=.1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def soft_dice_score(image1, image2, axis=(-3, -2, -1), eps=0.001):
    """Calculate average Dice across channels
    
    Args:
        image1, image2 (Tensor): The images to calculate Dice
        axis (tuple of int or int): The axes that the function sums across
        eps (float): Small number to prevent division by zero

    Returns:
        dice (float): The average Dice

    """
    #image2 = stable_one_hot(image2)
    intersection = K.sum(K.abs(image1 * image2), axis=axis)
    dices =  (2. * intersection + eps) / (K.sum(K.square(image1),axis) + K.sum(K.square(image2),axis) + eps)
    dice = K.mean(dices)
    return dice
    #intersection = K.sum(image1 * image2, axis=axis)
    #sum1 = K.sum(image1, axis=axis)
    #sum2 = K.sum(image2, axis=axis)
    #dices = 2 * (intersection + eps) / (sum1 + sum2 + eps)
    #dice = K.mean(dices)
    #return dice

def stable_one_hot(vec):
    """
    Args:
        vec: tf.Tensor, a batch of logits to be encoded
    
    Returns:
        tf.Tensor, a batch of numerically stable one-hot encoded logits
    """
    m = tf.math.reduce_max(vec, axis=1, keepdims=True)
    e = tf.math.exp(vec - m)
    mask = tf.cast(tf.math.not_equal(e, 1.0), tf.float32)
    vec -= 1e9 * mask
    return tf.nn.softmax(vec, axis=1)

def soft_dice_loss(y_true, y_pred):
    
    #print(y_true)
    #print(y_pred)
    return 1-soft_dice_score(y_true, y_pred)


def jaccard_score(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jaccard_distance(y_true, y_pred):
    return 1 - jaccard_score(y_true, y_pred)
