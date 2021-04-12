import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras

import config
from Dataset import generator
from model import Model


def train():
    # data
    Generator = generator()
    dataset = tf.data.Dataset.from_generator(
        generator=Generator,
        output_types=(tf.float32, tf.float32, tf.float32)
    )
    # model
    model = Model()
    # optim
    optimizer = keras.optimizers.Adam(lr=config.LR)
    # loss
    ...

    # training
    for i in range(config.START_EPOCH, config.MAX_EPOCH):
        for batch, (anchor, pos, neg) in enumerate(dataset):
            a_f, p_f, n_f = model(anchor, pos, neg)


if __name__ == '__main__':
    train()