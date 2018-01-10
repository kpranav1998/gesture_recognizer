# gesture_recognizer
This deep learning project detects simple gestures of hand
The aim of this project is to recognize simple gestures like STOP,PUNCH,PEACE,OK,NOTHING

The project is written in tensorflow,opencv,numpy and some python modules

CONTENTS
main.py:this file contains the model for training
run_saved_model.py:this file contains the pretrained model
test_data.txt:contains the test images
train_data.txt:contains the training images


PROCESS

#preprocessing part

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),2)   

th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#model

the layers are as follows

Convolution part:conv,relu,max_pool,conv,relu,max_pool'

Fully connected part:relu,relu,softmax




