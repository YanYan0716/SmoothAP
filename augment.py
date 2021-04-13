import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


import config


def orgAug(img):
    img = img[:, :, ::-1] / 255.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(config.IMG_SIZE, config.IMG_SIZE))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, size=(config.CROP_SIZE, config.CROP_SIZE, 3))
    return img


def posAug(img):
    img = img[:, :, ::-1] / 255.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(config.IMG_SIZE, config.IMG_SIZE))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, size=(config.CROP_SIZE, config.CROP_SIZE, 3))
    return img


def negAug(img):
    img = img[:, :, ::-1] / 255.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(config.IMG_SIZE, config.IMG_SIZE))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, size=(config.CROP_SIZE, config.CROP_SIZE, 3))
    return img


