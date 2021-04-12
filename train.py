import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras

import config
from Dataset import generator
from model import Model
from smoothAP import smoothAP


def train(dataset, model, loss, optimizer, scheduler):
    print('train ...')
    for i in range(config.START_EPOCH, config.MAX_EPOCH):
        for batch, (anchor, pos, neg) in enumerate(dataset):
            a_f, p_f, n_f = model(anchor, pos, neg)


def main():
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
    # scheduler
    scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LR,
        decay_steps=5,
        decay_rate=0.1,
    )
    # loss
    loss = smoothAP()

    # training
    train(
        dataset=dataset,
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler
    )


if __name__ == '__main__':
    train()