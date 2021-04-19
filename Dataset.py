import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import config
from augment import orgAug, posAug, negAug


def generatorT():
    """
    对应于triplet loss的数据，返回三份输出
    :return: anchor pos neg
    """
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

            anchor = orgAug(anchor)
            pos = posAug(pos)
            neg = negAug(neg)
            anchor_imgs.append(anchor)
            pos_imgs.append(pos)
            neg_imgs.append(neg)

            if len(anchor_imgs) == config.BATCH_SIZE:
                yield tf.cast(tf.convert_to_tensor(anchor_imgs), dtype=tf.float32), \
                      tf.cast(tf.convert_to_tensor(pos_imgs), dtype=tf.float32), \
                      tf.cast(tf.convert_to_tensor(neg_imgs), dtype=tf.float32),
                anchor_imgs = []
                pos_imgs = []
                neg_imgs = []
        except:
            continue


def generatorS():
    """
    对应于smoothAP loss的数据，返回一份输出
    :return: imgs
        格式：[
                class1_img1, class1_img2, class1_img3,
                class2_img1, class2_img2, class2_img3, ...
            ]
        每个类别的图片们都要放在一起，每个类别选多少张图片在config.SAMPLE_PER_CLASS中设置
    """
    img_file = pd.read_csv(config.IMG_DIR)
    imgs_list = img_file['name']
    imgs_label = img_file['label']
    all_num = len(imgs_list)
    index = np.arange(0, all_num)
    img_dict = {}
    for i in index:
        img_i_label = imgs_label[i]
        if img_i_label in img_dict.keys():
            img_dict[img_i_label].append(imgs_list[i])
        else:
            img_dict[img_i_label] = []
            img_dict[img_i_label].append(imgs_list[i])

    # shuffle
    np.random.shuffle(index)
    imgs = []
    for i in index:
        try:
            img_label = imgs_label[i]
            sample_imgs = img_dict[img_label]
            for j in range(0, config.SAMPLE_PER_CLASS):
                sample_index = np.random.randint(0, len(sample_imgs))
                sample_dir = os.path.join(config.ROOT_DIR, sample_imgs[sample_index])
                sample_img = cv2.imread(sample_dir)
                sample_img = orgAug(sample_img)
                imgs.append(sample_img)
            if len(imgs) == config.BATCH_SIZE:
                yield tf.cast(tf.convert_to_tensor(imgs), dtype=tf.float32)
                imgs = []
        except:
            continue


def testGenerator():
    dataset = tf.data.Dataset.from_generator(
        generator=generatorS,
        output_types=(tf.float32),
    )
    for i in range(1):
        for batch, img in enumerate(dataset):
            print(img.shape)
            break


if __name__ == '__main__':
    testGenerator()
    # sample_img = cv2.imread('D:\\algorithm\\CUB_200_2011\\images\\001.Black_footed_Albatross\\Black_Footed_Albatross_0051_796103.jpg')
    # sample_img = cv2.imread('.\\Black_Footed_Albatross_0051_796103.jpg')
    # sample_img = orgAug(sample_img)
    # print(sample_img.shape)