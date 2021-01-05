# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:40:29 2021

@author: bhat2
"""
# Python >= 3.8.5
# TensorFLow >= 1.15.0
# matplotlib >= 3.3.2

############################### Import Modules ###############################

import sys
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

print(' ')
print('Python: {}'.format(sys.version))
print('TensorFLow Version: ', tf.__version__)
print('Matplotlib Version: ', matplotlib.__version__)

###################### Importing and Preprocessing data #######################

mnist = tf.keras.datasets.fashion_mnist

# spliting data into training set and testing set
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# plotting an image from the training set to see how they look (e.g. index = 42, image of a boot)
plt.imshow(training_images[42])
# We can look at the values and the range for the image by printing an image on the terminal
# print(training_labels[42])
# print(training_images[42])

# in this dataset, there are 70,000 total images and each image is labeled.
# for example, the image of a 'boot' is labeled as '9'
# each image is 28 x 28 pixels and has 1 color channel (gray scale image)
# the pixel vales are in the range from 0 to 255

# we need to normalize the pixel values between 0 and 1 before feeding it to the model
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images  = training_images / 255.0
test_images = test_images / 255.0

####################### Model Building and Testing ############################
"""
# Model 1: Simple deep neural network model with 3 layers, without convolution
# designing the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# Sequential: That defines a SEQUENCE of layers in the neural network
# Flatten: our images were a square, when you printed it. Flatten just takes that square and turns it into a 1 dimensional set.
# Dense: Adds a layer of neurons. Each layer of neurons need an activation function to tell them what to do.
# Relu effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# Softmax takes a set of values, and effectively picks the biggest one

# compiling the model with an optmizer and loss function
model.compile(optimizer = tf.compat.v1.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# now we can fit the model with training data. here we are asking the model to
# fit the training images to their labels -- figuring out the relationship between 
# an image and its label. we can see how the accuracy increasing and loss 
# decreasingwith each epochs
model.fit(training_images, training_labels, epochs=10)
# in my case, this is the final result after training:
# epoch = 001 -- loss: 0.4889 - acc: 0.8230
# ...
# epoch = 100 -- loss: 0.0620 - acc: 0.9779

# evaluating the model using testing data that the model has ot seen before
model.evaluate(test_images, test_labels)
# my result: loss: 0.9511 - acc: 0.8969
"""
# Model 2: deep neural network model with convolution and pooling layers

# designing the model
# Step 1: gather the data. instead of feeding 60,000 images of 28 x 28 x 1 size
#         we feed a single 4D list that is 60000 x 28 x 28 x 1, and the same for the test images
#         this is the way to tell the convolution layer the shape of the incoming data
# Step 2: its a 2D convolutional layer of order 64 (64 random filers) where the filter size if (3 x 3)
#         the activation for this layer is relu, defined above
# Step 3: its a 2D max pooling layer to reduce the image size and retain the image information
# [add another convolutional layer followed by a pooling layer before feeding the data to the DNN]
# [the rest of it is Model 1]
model = tf.keras.models.Sequential([
                                   tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                   tf.keras.layers.MaxPooling2D(2, 2),
                                   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                   tf.keras.layers.MaxPooling2D(2,2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation='relu'),
                                   tf.keras.layers.Dense(10, activation='softmax') # there are 10 labels
                                 ])
# compiling the model with an optmizer and loss function
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# printing the summary of the model
model.summary()
# fitting the model using training data
model.fit(training_images, training_labels, epochs=2)
# evaluating the model using testing data that the model has ot seen before
model.evaluate(test_images, test_labels)

######################### Visualizing Image Processing ########################

# This code will show us the convolutions graphically. The print (test_labels[:100])
# shows us the first 100 labels in the test set, and you can see that the ones
# at index 0, index 23 and index 28 are all the same value (9). They're all shoes. 
# Let's take a look at the result of running the convolution on each, and you'll 
# begin to see common features between them emerge. Now, when the DNN is training
# on that data, it's working with a lot less, and it's perhaps finding a commonality
# between shoes based on this convolution/pooling combination.

FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28

plt.figure(figsize=(15, 7.5))
plt.imshow(test_images[FIRST_IMAGE])
plt.figure(figsize=(15, 7.5))
plt.imshow(test_images[SECOND_IMAGE])
plt.figure(figsize=(15, 7.5))
plt.imshow(test_images[THIRD_IMAGE])

f, axarr = plt.subplots(3,4)
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

###############################################################################
"""
Python: 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
TensorFLow Version:  1.15.0
Matplotlib Version:  3.3.2
Model: "sequential_11"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 26, 26, 64)        640       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 11, 11, 64)        36928     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten_11 (Flatten)         (None, 1600)              0         
_________________________________________________________________
dense_38 (Dense)             (None, 128)               204928    
_________________________________________________________________
dense_39 (Dense)             (None, 10)                1290      
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples
Epoch 1/2
60000/60000 [==============================] - 51s 857us/sample - loss: 0.4340 - acc: 0.8433
Epoch 2/2
60000/60000 [==============================] - 53s 882us/sample - loss: 0.2901 - acc: 0.8922
10000/10000 [==============================] - 3s 332us/sample - loss: 0.3113 - acc: 0.8852
"""
