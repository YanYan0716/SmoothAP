import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


import config


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = keras.applications.ResNet50(
            include_top=True,
            weights='imagenet',
            classes=1000
        )
        self.lase_linear = keras.layers.Dense(
            units=config.EMBED_DIM,
            activation=keras.activations.relu,
        )

    def call(self, x):
        y = self.base_model(x)
        y = self.lase_linear(y)
        return y

    def model(self):
        input = keras.Input(shape=(224, 224, 3), dtype=tf.float32)
        return keras.Model(inputs=input, outputs=self.call(input))


if __name__ == '__main__':
    a = np.random.normal(size=(3, 224, 224, 3))
    x = tf.convert_to_tensor(a)
    print(x.shape)
    net = Model().model()
    # net.save_weights(config.SAVE_PATH)
    net.load_weights(config.CONTINUE_PATH)
    # y = net(x)
    # print(y.shape)