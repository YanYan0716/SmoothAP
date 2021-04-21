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
        d[i] = np.linalg.norm(imgsFts[i, :] - computed_centroids[model_generated_cluster_labels[i], :])

    label_pred = np.zeros(len(imgsFts))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        label_pred[index] = cid

    N = len(target_labels)

    # 找出有多少个类别
    avali_labels = np.unique(target_labels)
    n_labels = len(avali_labels)

    # 计算真实类别分别属于每个预测类别的数量
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avali_labels[i])[0])

    keys = np.unique(label_pred)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()

    for i in range(keys):
        item_map.update([(keys[i], values[i])])

    # 预测结果中每个类别的数目
    count_item = np.zeros(num_item)
    for i in range(len(target_labels)):
        index = item_map[label_pred[i]]
        count_item[index] = count_item[index] + 1

    tp_fp = 0  # tp+fp
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    tp = 0  # tp
    for k in range(n_labels):
        member = np.where(target_labels == avali_labels[k])[0]
        member_ids = label_pred[member]
        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)
    # fp
    fp = tp_fp - tp
    # fn
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)
    fn = count - tp
    beta = 1
    P = tp / (tp+fp)
    R = tp / (tp+fn)
    F1 = (beta*beta + 1) * P * R / (beta*beta*P+R)
    return F1


def test(model, imgsPath, imgsLabel):
    """

    :param model:
    :param imgsPath:
    :param imgsLabel:
    :return: f1分数 召回率 特征向量
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
    return F1, recall_at_k, imgs_fts


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