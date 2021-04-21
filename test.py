import tensorflow as tf
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.special import comb
# from sklearn import met

from model import Model
import config


def readImg(imgPath):
    img = cv2.imread(imgPath)
    img = img[:, :, ::-1] / 255.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(config.IMG_SIZE, config.IMG_SIZE))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, size=(config.CROP_SIZE, config.CROP_SIZE, 3))
    return img


def f1_score(model_generated_cluster_labels, target_labels, imgsFts, computed_centroids):
    """
    多分类问题中的衡量指标，包括准确率和召回率，认为准确率和召回率同等重要
    公式：f1=2*(precision*recall)/(precision+recall) 数值越大越好 最大值为1，最小值为0
    :param model_generated_cluster_labels: 聚类后的标签
    :param target_labels: 真实标签
    :param feature_coll: 特征向量
    :param imgsFts: 聚类的中心向量
    :return:
    """
    d = np.zeros(len(imgsFts))
    for i in range(len(imgsFts)):
    return 0


def test(model, imgsPath, imgsLabel):
    """
    :param model:
    :param imgsPath:
    :param imgsLabel:
    :return:
    """
    target_labels, imgs_fts = [], []

    for imgpath in imgsPath:
        img = readImg(imgpath)
        img = tf.expand_dims(img, axis=0)
        img_ft = model(img)
        imgs_fts.append(img_ft)

    imgs_fts = np.vstack(imgs_fts).astype(np.float32)
    target_labels = np.hstack(imgsLabel).reshape(-1, 1)

    # 使用kmeans对model的输出进行聚类
    kmeans = KMeans(n_clusters=2, random_state=0).fit(imgs_fts)
    model_generated_cluster_labels = kmeans.labels_  # 聚类后的标签
    computed_centroids = kmeans.cluster_centers_  # 聚类后的类中心

    # NMI = metrics.cluster.normalized_mutual_info_score(
    #     model_generated_cluster_labels.reshape(-1),
    #     target_labels.reshape(-1)
    # )

    k_closest_points = squareform(pdist(imgs_fts)).argsort(1)[:, :int(np.max(config.K_VALS)+1)]  # 得到距离最近的前k个数的index
    k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]  #找到k_closet_points对应的类别

    recall_all_k = []
    for k in config.K_VALS:
        recall_at_k = np.sum([1 for target, recalled_pred in zip(target_labels, k_closest_classes)
                               if target in recalled_pred[:k]]) / len(target_labels)
        recall_all_k.append(recall_at_k)

    # 计算f1 score
    F1 = f1_score(model_generated_cluster_labels, target_labels, imgs_fts, computed_centroids)
    return 0


def main():
    model = Model().model()
    # model.load_weights(filepath=config.LOAD_PATH)
    model.trainable = False

    # dataset
    imgs_path = pd.read_csv('label4000.csv')['name']
    imgs_label = pd.read_csv('label4000.csv')['label']

    # test
    result = test(model, imgs_path, imgs_label)


if __name__ == '__main__':
    main()