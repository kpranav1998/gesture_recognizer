import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2

num_classes = 5


# layer 1


# initialisers

TEST_DATA='/home/kpranav1998/PycharmProjects/gesture_recognizer/test_data.txt'

x_test, y_test = image_preloader(TEST_DATA, mode='file', image_shape=(64, 64), categorical_labels=True, normalize=True,
                                 grayscale=True)
x_test = np.reshape(x_test, [len(x_test), 64, 64, 1])
print(x_test.shape)
print(x_test.dtype)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph('model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    y_predicted=graph.get_tensor_by_name("y_predicted:0")
    x=graph.get_tensor_by_name("x:0")
    y=graph.get_tensor_by_name("y:0")




    ans = sess.run(y_predicted, feed_dict={x: x_test, y: y_test})
    ans2 = np.argmax(ans, 1)

    answer = np.reshape(ans2, newshape=[len(x_test), 1])
    y_test_temp = np.argmax(y_test, 1)
    y_test_temp = np.reshape(y_test_temp, newshape=[len(x_test), 1])
    answer = np.hstack((answer, y_test_temp))
    print(answer)