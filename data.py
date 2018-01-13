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

TRAIN_DATA='/home/kpranav1998/PycharmProjects/gesture_recognizer/imgfolder_b/'

fr = open('train_data.txt', 'w')
files=[]
file_names=os.listdir(TRAIN_DATA)
for filename in tqdm(file_names):

    if(filename.find("iiiok")!=-1):
        path = os.path.join(TRAIN_DATA, filename)
        files.append([path, ' 0'])
    elif(filename.find("nnnothing")!=-1):
           path = os.path.join(TRAIN_DATA, filename)
           files.append([path, ' 1'])
    elif (filename.find("pppeace")!=-1):
        path = os.path.join(TRAIN_DATA, filename)
        files.append([path, ' 2'])
    elif (filename.find("pppunch")!=-1):
        path = os.path.join(TRAIN_DATA, filename)
        files.append([path, ' 3'])
    elif (filename.find("ssstop")!=-1):
        path = os.path.join(TRAIN_DATA, filename)
        files.append([path, ' 4'])



file_names=os.listdir(TRAIN_DATA)


random.shuffle(files)

for file in files:
    fr.write(file[0]+file[1]+'\n')



