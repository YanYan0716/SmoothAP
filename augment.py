import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


import config


def orgAug(img):
    img = tf.image.resize(img, size=config.IMG_SIZE)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_flip_left_right(img, )