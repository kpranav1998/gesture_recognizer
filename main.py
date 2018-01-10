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


##########directories##########
TRAIN_DATA='/home/kpranav1998/PycharmProjects/gesture_recognizer/train_data.txt'

##################

##########image loading##############
x_train,y_train=image_preloader(TRAIN_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
x_train=np.reshape(x_train,[len(x_train),64,64,1])
print("1")
##########
x=tf.placeholder(shape=[None,64,64,1],dtype=tf.float32,name='x')
y=tf.placeholder(shape=[None,5],dtype=tf.float32,name='y')

print("2")

######       hyperparameters    ###########
beta1=0.9
beta2=0.99
learn_rate=0.000001
epochs = 200
batch_size=16
no_itr_per_epoch=len(x_train)//batch_size
saver = tf.train.Saver()
print("3")
###########  end ######


############# model##############

filter1= tf.get_variable("filter1",shape=[5,5,1,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
filter2 = tf.get_variable("filter2",shape=[3,3,64,32],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

conv1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME',name="conv1")
relu1=tf.nn.relu(conv1,name='relu1')
pool1=out_layer1=tflearn.layers.conv.max_pool_2d(relu1, kernel_size=5)
conv2=tf.nn.conv2d(input=pool1,filter=filter2,strides=[1,1,1,1],padding='SAME',name="conv2")
relu2=tf.nn.relu(conv2,name='relu2')
pool2=tflearn.layers.conv.max_pool_2d(relu2, kernel_size=3)
flat_array=tflearn.layers.core.flatten(pool2,name="flat_array")
fc1 = tflearn.layers.core.fully_connected(flat_array,
    4096,
    activation='relu',weights_init=tflearn.initializations.xavier (uniform=True, seed=None, dtype=tf.float32),name='fc1'
)

fc2 = tflearn.layers.core.fully_connected(fc1,
    5,
    activation='relu',weights_init=tflearn.initializations.xavier (uniform=True, seed=None, dtype=tf.float32),name='fc1'
)
y_predicted = tf.nn.softmax(logits=fc2,name='y_predicted')

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


