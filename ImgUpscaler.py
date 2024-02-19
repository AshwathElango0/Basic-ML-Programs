import tensorflow as tf
from tensorflow import keras 
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import torch
import cv2 as cv

(train_img, _), (test_img, _) = keras.datasets.mnist.load_data()        #loading data

train_img = torch.tensor(train_img)
test_img = torch.tensor(test_img)

inimg = torch.cat((train_img, test_img), dim=0)             #combining all images into one tensor
inimg = np.expand_dims(inimg, axis=-1)
inimg = inimg.astype('float32')/255.0                        #normalizing

model = keras.Sequential()                                      #building neural network for upscaling
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])           #network is untrained, so upscaling is inefficient




x = 5                                        #index of image to upscale
def upscale():
    img = inimg[x]
    img = np.expand_dims(img, axis=0)
    upimg = model.predict(img)
    upimg = (upimg * 255).astype('uint8')
    upimg = cv.cvtColor(upimg, cv.COLOR_GRAY2BGR)       #need to convert to color as original images had no color channels
    upimg = cv.cvtColor(upimg, cv.COLOR_BGR2GRAY)        #restoring to gray now that channels have been added  
    upimg = cv.resize(upimg, None, fx=30, fy=450, interpolation=cv.INTER_LINEAR)   #enlarging
    cv.imshow('Better image', upimg)            #display
    cv.waitKey(0)
    cv.destroyAllWindows()

upscale()
    