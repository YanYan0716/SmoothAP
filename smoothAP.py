import tensorflow as tf
import tensorflow.keras as keras


import config


def sigmoid(x, temp=1.0):
    exponent = -x / temp
    exponent = tf.clip_by_value(exponent, clip_value_min=-50, clip_value_max=50)
    y = 1.0 / (1.0 + tf.exp(exponent))
    return y


def compute_aff(x):
    return tf.


def smoothAP():
    mask = 1.0 - tf.eye(config.BATCH_SIZE)
    mask =