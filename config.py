import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset
ROOT_DIR = 'D:\\algorithm\\CUB_200_2011\\images\\'
IMG_DIR = 'D:\\algorithm\\CUB_200_2011\\aaa.csv'
BATCH_SIZE = 4
IMG_SIZE = 256
CROP_SIZE = 224
SAMPLE_PER_CLASS = 4

# model
EMBED_DIM = 256

# train
MAX_EPOCH = 100
START_EPOCH = 0
LOG_EPOCH = 30

# optimizer
LR = 0.01

# loss
ANNEAL = 0.01

# test
LOAD_PATH = './weights/best'
K_VALS = [1, 4, 16, 32]
FC_LR_MUL = 1

