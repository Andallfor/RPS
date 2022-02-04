#tar.gz is monkEEEEEEEE :() :() water monkE moment 
import tensorflow as tf
from keras import layers, models, datasets, metrics
import matplotlib as plt
import numpy as np
#import cv2 as opencv
from collections import Counter
import ssl
import sys
import datetime

checkpoint_path = "training_1/cp.ckpt"

#Get the dataset ready monke :() :()
ssl._create_default_https_context = ssl._create_unverified_context
(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data() #fine_grained labels, classes not superclasses

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

training_img = training_img / 255 #normalizing pixel values https://m.youtube.com/watch?v=T78nq62aQgM
testing_img = testing_img / 255
number_of_classes = 10

#The model
model = tf.keras.models.Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(300, activation = 'relu')) #extra layer cause accuracy is garbage
model.add(layers.Dense(300, activation = 'relu'))
model.add(layers.Dense(10)) #output?

#sets up tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#trains the model
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(training_img, training_label, epochs = 10, validation_data=(testing_img, testing_label), callbacks=[tensorboard_callback]) #increase epoch length later

#saves final version checkpoint if prompted to
savemanual = input("save trained weights?")
if savemanual == "y" or savemanual == "Y":
  model.save(checkpoint_path)
elif savemanual == "n" or savemanual == "N":
  print("RIP this model")
  sys.exit()
else:
  print("invalid input, please type y or n")
  sys.exit()
