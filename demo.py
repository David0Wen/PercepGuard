import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tensorflow.keras.models import load_model

trans_mat = np.array([
    [.5, 0, -1, 0],
    [0, .5, 0, -1],
    [.5, 0, 1, 0],
    [0, .5, 0, 1]])

cls = 'car' # or 'ped'

## load the pretrain LSTM model
lstm_cls = load_model('./models/bdd100k', compile=False)

## load data samples and preprocess them
seq = np.genfromtxt('./data/bdd100k_{}.csv'.format(cls), delimiter=',')
seq = np.matmul(seq[:, :4] / ([1280, 720] * 2), trans_mat)

bbox_seq = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))
obj_cls = lstm_cls(bbox_seq)
global_init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(global_init)
    c = sess.run(obj_cls, feed_dict={bbox_seq:seq[None, ...]})
    print(c) # [bikes, buses, cars, pedestrians, and trucks]
