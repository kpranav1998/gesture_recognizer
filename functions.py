import tensorflow as tf
from tflearn.data_utils import image_preloader
import numpy as np
import os

def forward_propagation_initial(x,parameter):
    ######loading parameters############
    filter1=parameter["filter1"]
    filter2 = parameter["filter2"]
    filter1 = parameter["filter1"]


    ########end loading

    ##propagation

    conv1=tf.nn.conv2d(input=x,filter=64,strides=[1,1,1,1],padding="SAME",name="conv1")
    relu1=tf.nn.relu(conv1)
    pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1])
    conv2 = tf.nn.conv2d(input=pool1, filter=32, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1])
    flat_array = tf.contrib.layers.flatten(pool2)
    #fully_connected layer1

    fc1=tf.contrib.layers.fully_connected(
    inputs=flat_array,
    num_outputs=4096,
    activation_fn=tf.nn.relu(),
    normalizer_fn=tf.nn.batch_normalization(),
    weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
    weights_regularizer=tf.nn.dropout(),
    biases_initializer=tf.zeros_initializer()
    )

    fc2=tf.contrib.layers.fully_connected(
            inputs=flat_array,
            num_outputs=64,
            activation_fn=tf.nn.relu(),
            normalizer_fn=tf.nn.batch_normalization(),
            weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
            weights_regularizer=tf.nn.dropout(),
            biases_initializer=tf.zeros_initializer()

    )
    return fc2





def forward_propogation(x,parameters):
    pa


    relu1=tf.nn.relu(conv1)
    pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1])
    relu2 = tf.nn.relu(conv2)
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1])
    flat_array = tf.contrib.layers.flatten(pool2)-













