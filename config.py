import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset
# ROOT_DIR = ''
# IMG_DIR = 'label4000.csv'
ROOT_DIR = '../input/cub-200-2011/CUB_200_2011/images'
IMG_DIR = '../input/cub-200-2011/CUB_200_2011/train.csv'
BATCH_SIZE = 64
IMG_SIZE = 256
CROP_SIZE = 224
SAMPLE_PER_CLASS = 4
N_CLASSES = 200

# model
EMBED_DIM = 256

# train
MAX_EPOCH = 100
START_EPOCH = 0
LOG_EPOCH = 6
SAVE_PATH = './weights/W'


# optimizer
LR = 1e-5

# loss
ANNEAL = 0.01

# test
LOAD_PATH = './weights/best'
K_VALS = [1, 4, 16, 32]



FC_LR_MUL = 1

