import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

# dataset
BATCH_SIZE = 32
IMG_SIZE = 256
CROP_SIZE = 224


K_VALS = [1, 4, 16, 32]
FC_LR_MUL = 1

# model
EMBED_DIM = 512