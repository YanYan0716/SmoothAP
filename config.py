import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset
IMG_DIR = './label4000.csv'
BATCH_SIZE = 8
IMG_SIZE = 256
CROP_SIZE = 224
SAMPLE_PER_CLASS = 4

# model
EMBED_DIM = 256

# train
MAX_EPOCH = 100
START_EPOCH = 0

# optimizer
LR = 0.01

# loss
ANNEAL = 0.01
NUM_ID = 4



K_VALS = [1, 4, 16, 32]
FC_LR_MUL = 1

