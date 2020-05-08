import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy

def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    loss = 1. - 2 * intersection / (union + K.epsilon())
    return loss

def ce_dice_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    dice_loss = - tf.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + dice_loss
    return loss


def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    loss = 1. - intersection / (union + K.epsilon())
    return loss


def ce_jaccard_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = - tf.log((intersection + K.epsilon()) / (union + K.epsilon()))
    loss = ce_loss + jaccard_loss
    return loss


def tversky_loss(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return 1 - (true_pos + K.epsilon())/(true_pos + alpha * false_neg + (1-alpha) * false_pos + K.epsilon())


