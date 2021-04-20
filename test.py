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


def f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    return 0


def test(model, imgsPath):
    target_labels, imgs_fts = [], []

    for imgpath in imgsPath:
        img = readImg(imgpath)
        img = tf.expand_dims(img, axis=0)
        img_ft = model(img)
        imgs_fts.append(img_ft)

    imgs_fts = np.vstack(imgs_fts).astype(np.float32)
    kmeans = KMeans(n_clusters=config.N_CLASSES, random_state=0).fit(imgs_fts)
    model_generated_cluster_labels = kmeans.labels_
    computed_centroids = kmeans.cluster_centers_

    NMI = metrics.cluster.normalized_mutual_info_score(
        model_generated_cluster_labels.reshape(-1),
        target_labels.reshape(-1)
    )

    k_closest_points = squareform(pdist(imgs_fts)).argsort(1)[:, :int(np.max(config.K_VALS)+1)]
    k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

    recall_all_k = []
    for k in config.K_VALS:
        recall_at_k = np.sum([1 for target, recalled_pred in zip(target_labels, k_closest_classes)
                               if target in recalled_pred[:k]]) / len(target_labels)
        recall_all_k.append(recall_at_k)

    F1 = f1_score(model_generated_cluster_labels, target_labels, imgs_fts, computed_centroids)
    return 0


def main():
    model = Model()
    # model.load_weights(filepath=config.LOAD_PATH)
    model.trainable = False

    # dataset
    imgs_path = pd.read_csv('./label4000.csv')['name']

    # test
    result = test(model, imgs_path)


if __name__ == '__main__':
    main()