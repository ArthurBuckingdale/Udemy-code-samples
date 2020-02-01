# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:09:33 2019

@author: ryanc
"""

#Part 1 - BUilding the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
tf_config = tf.ConfigProto(allow_soft_placement = False)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)



#initialising the CNN
classifier = Sequential()

#step 1 - Convolution 
classifier.add(Conv2D(32,3,3, input_shape=(64,64,3), activation = 'relu'))

#Step two Pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))
#adding a second convolutional layer

classifier.add(Conv2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#adding a third convolutional layer
classifier.add(Conv2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step three Flattening
classifier.add(Flatten())

#step four full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#part 2 fitting images to the CNN 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size= (64, 64),
            batch_size=512,
            class_mode='binary')

test_set = test_datagen.flow_from_directory(
                        'dataset/test_set',
                        target_size=(64, 64),
                        batch_size=512,
                        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/512,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/512,
        workers = 16)

