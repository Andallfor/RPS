#tar.gz is monkEEEEEEEE :() :() water monkE moment 
import tensorflow as tf
from keras import layers, models, datasets, metrics
import matplotlib as plt
import numpy as np
#import cv2 as opencv
from collections import Counter
import ssl
import sys

checkpoint_path = "training_1/cp.ckpt"

#Get the dataset ready monke :() :()
ssl._create_default_https_context = ssl._create_unverified_context

(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data() #fine_grained labels, classes not superclasses
training_img = training_img / 255 #normalizing pixel values https://m.youtube.com/watch?v=T78nq62aQgM
testing_img = testing_img / 255
number_of_classes = 10

#reshaping dataset
assert training_img.shape == (50000, 32, 32, 3)
assert testing_img.shape == (10000, 32, 32, 3)
assert training_label.shape == (50000, 1)
assert testing_label.shape == (10000, 1)

#The model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(300, activation = 'relu')) #extra layer cause accuracy is garbage
model.add(layers.Dense(300, activation = 'relu'))
model.add(layers.Dense(10)) #output?

#trains the model
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(training_img, training_label, epochs = 1, validation_data=(testing_img, testing_label)) #increase epoch length later

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

"""
#applies the model over each image of the dataset
predictions = [model.predict(training_img[n]) for n in range(0, len(training_img))]
#for n in range(0, len(predictions)):
	#prediction[n] = np.argmax(predictions[n], axis = 1)
#validation_predictions = [model.predict(testing_img[n] for n in range(0, len(testing_image)))]

for n in range(0, len(predictions)):
    validation_predictions[n] = np.argmax(validation_predictions[n], axis = 1)

model_1_predictor = tf.keras.models.Sequential()
model_1_predictor_x.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
model_1_predictor_x.add(layers.MaxPooling2D((2,2)))
model_1_predictor_x.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_1_predictor_x.add(layers.MaxPooling2D((2,2)))
model_1_predictor_x.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_1_predictor_x.add(layers.Flatten())
model_1_predictor_x.add(layers.Dense(64, activation = 'relu'))
model_1_predictor_x.add(layers.Dense(2))
model_1_predictor_x.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), metrics = ["accuracy"])
implementation_predictor = model_1_predictor.fit(training_img, predictions, epochs = 2, validation_data = (testing_img, validation_predictions))

model.summary()

#Implement the model
#Note: Create a single function for creating the model. Don't go through this every time. Return the model.
"""
