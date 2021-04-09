import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

import config


def generator(img_list):
    all_num = len(img_list)
    index = np.arange(0, all_num)

    while True:
        # shuffle
        np.random.shuffle(index)
        count = 0
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
                    yield anchor, pos_imgs, neg_imgs
                    anchor_imgs = []
                    pos_imgs = []
                    neg_imgs = []
            except:
                print('something wrong in data generator ...')
                continue


if __name__ == '__main__':
    img_list = pd.read_csv('./label4000.csv')['name']
    print(img_list[1])
    print(len(img_list))

    Generator = generator(img_list)