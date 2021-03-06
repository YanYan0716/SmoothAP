import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import config
from Dataset import generatorS
from model import Model
from smoothAP import smoothAP


def train(dataset, model, criterion, optimizer, scheduler):
    print('train ...')
    avgloss = 0
    BESTloss = 0.60

    for epoch in range(config.START_EPOCH, config.MAX_EPOCH):
        for batch, imgs in enumerate(dataset):
            with tf.GradientTape() as tape:
                fts = model(imgs)
                loss = criterion(fts)
                avgloss += loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if (batch + 1) % config.LOG_EPOCH == 0:
                avgloss = avgloss / config.LOG_EPOCH
                if BESTloss > avgloss:
                    model.save_weights(config.SAVE_PATH)
                    BESTloss = avgloss
                    print(f'saved model to {config.SAVE_PATH}')
                print(f'[max_epoch: %3d]' % config.MAX_EPOCH + ',[epoch:%3d/' % (epoch + config.START_EPOCH)
                        + 'idx: %3d]' % batch + '[Loss:%.4f' % (avgloss) + '/ Best loss: %.4f]' % (BESTloss))
                avgloss = 0
        scheduler.__call__(step=epoch)


def main():
    # data
    dataset = tf.data.Dataset.from_generator(
        generator=generatorS,
        output_types=(tf.float32),
    )
    # model
    model = Model().model()
    if config.CONTINUE:
        print('loading weights ...')
        model.load_weights(config.CONTINUE_PATH)
    # optim
    optimizer = keras.optimizers.Adam(lr=config.LR)
    # scheduler
    scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.LR,
        decay_steps=5,
        decay_rate=0.1,
    )
    # loss
    loss = smoothAP

    # training
    train(
        dataset=dataset,
        model=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler
    )


if __name__ == '__main__':
    main()
