# Convolution Neural Network

# Importing Libraries

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
# Initialize the CNN


CNN_Model = Sequential()

# Adding Layers

# 1 - Convolution Layer
CNN_Model.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# 2 Pooling Layer
CNN_Model.add(MaxPooling2D(pool_size=[2, 2]))

# Flatten Layer
CNN_Model.add(Flatten())

# Fully Connected Layer
CNN_Model.add(Dense(units=128, activation='relu'))
CNN_Model.add(Dense(units=1, activation='sigmoid'))

# Model Compile
CNN_Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Preprocessing

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_Set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Fitting the CNN

CNN_Model.fit_generator(
    training_Set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000)
