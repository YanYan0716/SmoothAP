import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


import config


def sigmoid(x, temp=1.0):
    exponent = -x / temp
    exponent = tf.clip_by_value(exponent, clip_value_min=-50, clip_value_max=50)
    y = 1.0 / (1.0 + tf.exp(exponent))
    return y


def compute_aff(x):
    return tf.matmul(x, tf.transpose(x))


def smoothAP(pred_fts):
    mask = 1.0 - tf.eye(config.BATCH_SIZE)
    mask = tf.repeat(input=tf.expand_dims(mask, axis=0), repeats=config.BATCH_SIZE, axis=0)

    sim_all = compute_aff(pred_fts)
    sim_all_repeat = tf.repeat(tf.expand_dims(sim_all, axis=1), repeats=config.BATCH_SIZE, axis=1)
    sim_diff = sim_all_repeat - tf.transpose(sim_all_repeat, perm=[0, 2, 1])
    sim_sg = sigmoid(sim_diff, temp=config.ANNEAL) * mask
    sim_all_rk = tf.reduce_sum(sim_sg, axis=-1) + 1

    xs = tf.reshape(pred_fts, shape=(config.NUM_ID, int(config.BATCH_SIZE/config.NUM_ID), config.EMBED_DIM))
    pos_mask = 1.0 - tf.eye(int(config.BATCH_SIZE/config.NUM_ID))
    pos_mask = tf.expand_dims(tf.expand_dims(pos_mask, axis=0), axis=0)
    pos_mask = tf.repeat(pos_mask, repeats=int(config.BATCH_SIZE/config.NUM_ID), axis=1)
    pos_mask = tf.repeat(pos_mask, repeats=config.NUM_ID, axis=0)

    sim_pos = tf.matmul(xs, tf.transpose(xs, perm=[0, 2, 1]))
    sim_pos_repeat = tf.repeat(tf.expand_dims(sim_pos, axis=2), repeats=int(config.BATCH_SIZE/config.NUM_ID), axis=2)
    sim_pos_diff = sim_pos_repeat - tf.transpose(sim_pos_repeat, perm=[0, 1, 3, 2])
    sim_pos_sg = sigmoid(sim_pos_diff, temp=config.ANNEAL) * pos_mask
    sim_pos_rk = tf.reduce_sum(sim_pos_sg, axis=-1) + 1

    ap = tf.zeros(1)
    group = int(config.BATCH_SIZE / config.NUM_ID)
    for ind in range(config.NUM_ID):
        pos_divide = tf.math.add(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
        ap = ap + ((pos_divide / group) / config.BATCH_SIZE)
    return (1-ap)

    return pos_mask


def test():
    x = tf.random.normal(shape=(config.BATCH_SIZE, config.EMBED_DIM))
    y = smoothAP(x)
    print(y.shape)


if __name__ == '__main__':
    test()