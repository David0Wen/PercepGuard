import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tensorflow.keras.models import load_model

frame_size = [1280, 720]

trans_mat = np.array([
    [.5, 0, -1, 0],
    [0, .5, 0, -1],
    [.5, 0, 1, 0],
    [0, .5, 0, 1]])

class_map = ['bike', 'bus', 'car', 'ped', 'truck']

## load the pretrain LSTM model
lstm_cls = load_model('./models/bdd100k', compile=False)

bbox_seq = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))
obj_cls = lstm_cls(bbox_seq)
global_init = tf.global_variables_initializer()

check_attack = lambda condition: "No Attack Detected" if condition else "Alarm! Attack Detected"

# The input file path (containing boundingboxs seq and predicted class label)
file_path = 'test_input.csv'
# Read the file directly to understand its structure
with open(file_path, 'r') as file:
    lines = file.readlines()

    # Parsing the 9x4 array and the scalar
    array_data = [list(map(float, line.split(','))) for line in lines[:-1]]
    predLabel = int(lines[-1].strip())  # The last line contains the scalar

    seq = np.array(array_data)
    seq = np.matmul(seq[:, :4] / (frame_size * 2), trans_mat)

with tf.Session() as sess:
    sess.run(global_init)

    # cls = class_map[predLabel]
    ## load data samples and preprocess them
    # seq = np.genfromtxt('./data/bdd100k_{}.csv'.format(cls), delimiter=',')
    c = sess.run(obj_cls, feed_dict={bbox_seq:seq[None, ...]})
    index_of_max = np.argmax(c)
    print(c) # [bikes, buses, cars, pedestrians, and trucks]
    print("Detected class: ", class_map[predLabel])
    print("Inference from bounding boxes: ", class_map[index_of_max])
    print(check_attack(predLabel == index_of_max))
