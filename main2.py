import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
#for writing text files
import glob
import os
import random
#reading images from a text file
from tflearn.data_utils import image_preloader
from tqdm import tqdm
import math



##########image loading##############
TRAIN_DATA='/home/kpranav1998/PycharmProjects/gesture_recognizer/train_data.txt'

x_train,y_train=image_preloader(TRAIN_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
x_train=np.reshape(x_train,[len(x_train),64,64,1])
print("1")
##########
######       hyperparameters    ###########
beta1=0.9
beta2=0.99
learn_rate=0.000001
epochs = 1
batch_size=16
no_itr_per_epoch=len(x_train)//batch_size
saver = tf.train.Saver()
print("3")
##############
x=tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
y=tf.placeholder(shape=[None,5],dtype=tf.float32)



input_layer=x
#convolutional layer 1 --convolution+RELU activation
conv_layer1=tflearn.layers.conv.conv_2d(input_layer, nb_filter=64, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

#2x2 max pooling layer
out_layer1=tflearn.layers.conv.max_pool_2d(conv_layer1, 2)


#second convolutional layer
conv_layer2=tflearn.layers.conv.conv_2d(out_layer1, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer2=tflearn.layers.conv.max_pool_2d(conv_layer2, 2)
# third convolutional layer
conv_layer3=tflearn.layers.conv.conv_2d(out_layer2, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')
out_layer3=tflearn.layers.conv.max_pool_2d(conv_layer3, 2)

#fully connected layer1
fcl= tflearn.layers.core.fully_connected(out_layer3, 4096, activation='relu' , name='FCL-1')
fcl_dropout_1 = tflearn.layers.core.dropout(fcl, 0.8)
#fully connected layer2
fc2= tflearn.layers.core.fully_connected(fcl_dropout_1, 4096, activation='relu' , name='FCL-2')
fcl_dropout_2 = tflearn.layers.core.dropout(fc2, 0.8)
#softmax layer output
y_predicted = tflearn.layers.core.fully_connected(fcl_dropout_2, 5, activation='softmax', name='output')

print("4")
##cost and optimizer#######
cost=tf.reduce_mean(-tf.multiply(y,tf.log(y_predicted))+tf.multiply((y-1),tf.log(1-y_predicted)))
optimizer=tf.train.AdamOptimizer(learning_rate=learn_rate,beta2=beta2,beta1=beta1).minimize(cost)

###acccuracy
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_predicted,1)),dtype=tf.float32))


#####initializer and saver########
init=tf.global_variables_initializer()
saver=tf.train.Saver()
save_path='model.ckpt'


###########mini_batches###########

with tf.Session() as sess:
    print("5")
    tf.global_variables_initializer().run()
    for iteration in range(epochs):
        print("Iteration no: {} ".format(iteration))

        previous_batch = 0
        # Do our mini batches:
        for i in range(no_itr_per_epoch):
            current_batch = previous_batch + batch_size
            x_input = x_train[previous_batch:current_batch]
            x_images = np.reshape(x_input, [batch_size, 64, 64,1])

            y_input = y_train[previous_batch:current_batch]
            y_label = np.reshape(y_input, [batch_size, 5])
            previous_batch = previous_batch + batch_size

            _, loss= sess.run([optimizer, cost], feed_dict={x: x_images, y: y_label})
            print(loss)
    save_path = saver.save(sess, "/home/kpranav1998/PycharmProjects/gesture_recognizer/model.ckpt")
    acc=sess.run(accuracy,feed_dict={x:x_train,y:y_train})
    print(acc*100)
