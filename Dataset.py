import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


import config


def generator():
    img_list = pd.read_csv(config.IMG_DIR)['name']
    all_num = len(img_list)
    index = np.arange(0, all_num)

    # shuffle
    np.random.shuffle(index)
    anchor_imgs = []
    pos_imgs = []
    neg_imgs = []

    for i in index:
        try:
            anchor = cv2.imread(img_list[i])
            if anchor is None:
                continue
            pos = cv2.imread(img_list[i])
            neg_index = np.random.randint(0, all_num)
            neg = cv2.imread(img_list[neg_index])

            anchor_imgs.append(anchor[:, :, ::-1].astype(np.float32))
            pos_imgs.append(pos[:, :, ::-1].astype(np.float32))
            neg_imgs.append(neg[:, :, ::-1].astype(np.float32))

            if len(anchor_imgs) == config.BATCH_SIZE:
                yield tf.cast(tf.convert_to_tensor(anchor_imgs), dtype=tf.float32), \
                      tf.cast(tf.convert_to_tensor(pos_imgs), dtype=tf.float32), \
                      tf.cast(tf.convert_to_tensor(neg_imgs), dtype=tf.float32),
                anchor_imgs = []
                pos_imgs = []
                neg_imgs = []
        except:
            continue


if __name__ == '__main__':
    imgs = []
    img = cv2.imread('./1.jpg')
    img = img[:, :, ::-1] / 255.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(224, 224))
    imgs.append(img)
    imgs.append(img)
    imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
    print(imgs.shape)
    plt.imshow(imgs[0])
    plt.show()

    # Generator = generator()
    # dataset = tf.data.Dataset.from_generator(
    #     generator=generator,
    #     output_types=(tf.float32, tf.float32, tf.float32)
    # )
    # for i in range(1):
    #     for batch, (anchor, pos, neg) in enumerate(dataset):
    #         print(anchor.shape)
    #         print(pos.shape)
    #         print(neg.shape)
    #         print('----------')
    #         break