import tensorflow as tf
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
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


def test(model, imgsPath):
    target_labels, imgs_fts = [], []

    for imgpath in imgsPath:
        img = readImg(imgpath)
        img = tf.expand_dims(img, axis=0)
        img_ft = model(img)
        imgs_fts.append(img_ft)

    imgs_fts = np.vstack(imgs_fts).astype(np.float32)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(imgs_fts)
    model_generated_cluster_labels = kmeans.labels_
    computed_centroids = kmeans.cluster_centers_
    print(model_generated_cluster_labels)
    print(computed_centroids.shape)
    # model_generated_cluster_labels = kmeans.labels_
    # computed_cenroids = kmeans.cluster_centers_
    #
    # NMI = metrics.cluster.normalized_mutual_info_score(
    #     model_generated_cluster_labels.reshape(-1),
    #     target_labels
    # )

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