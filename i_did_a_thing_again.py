import tensorflow as tf
from keras import layers, models, datasets
import matplotlib as plt
import numpy as np
import cv2 as opencv
from collections import Counter

(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data()

training_img = training_img / 255
testing_img = testing_img / 255

models = [0,0,0,0,0,0,0,0]
implementations = [0,0,0,0,0,0,0,0]
number_of_classes = 10

def create_model(model_number, blur_level):

	training_img_2, testing_img_2 = np.array([opencv.blur(img, (5,5), blur_level) for img in training_img]),  np.array([opencv.blur(img, (5,5), blur_level) for img in testing_img])

	models[model_number] = tf.keras.models.Sequential()
	models[model_number].add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
	models[model_number].add(layers.MaxPooling2D((2,2)))
	models[model_number].add(layers.Conv2D(64, (3, 3), activation='relu'))
	models[model_number].add(layers.MaxPooling2D((2,2)))
	models[model_number].add(layers.Conv2D(64, (3, 3), activation='relu'))
	models[model_number].add(layers.Flatten())
	models[model_number].add(layers.Dense(64, activation = 'relu'))
	models[model_number].add(layers.Dense(10))
	models[model_number].compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), metrics = ["accuracy"])
	implementations[model_number] = models[model_number].fit(training_img_2, training_label, epochs = 8, validation_data=(testing_img_2, testing_label), verbose = 2)

def create_predictor(base_model):

	model_number = base_model

	#Labels from first model
	predictions = models[base_model].predict(training_img)
	validation_predictions = models[base_model].predict(testing_img)
	predictions = predictions.tolist()
	validation_predictions = validation_predictions.tolist()
	n = 0
	for element in predictions:
		predictions[n] = [element.index(max(element))]
		n += 1

	n_2 = 0
	for element in validation_predictions:
		validation_predictions[n_2] = [element.index(max(element))]
		n_2 += 1


	label_list = training_label.tolist()
	validation_label_list = testing_label.tolist()

	training_prediction_labels = []
	validation_prediction_labels = []

	n_3 = 0
	for element in predictions:
		label = label_list[n_3]
		if element[0] == label[0]:
			training_prediction_labels.append([1])
		else:
			training_prediction_labels.append([0])
		n_3 += 1

	n_4 = 0
	for element in validation_predictions:
		label = validation_label_list[n_4]
		if element[0] == label[0]:
			validation_prediction_labels.append([1])
		else:
			validation_prediction_labels.append([0])
		n_4 += 1


	training_prediction_labels = np.array(training_prediction_labels)
	validation_prediction_labels = np.array(validation_prediction_labels)

	#Predictor model
	models[model_number+4] = tf.keras.models.Sequential()
	models[model_number+4].add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
	models[model_number+4].add(layers.MaxPooling2D((2,2)))
	models[model_number+4].add(layers.Conv2D(64, (3, 3), activation='relu'))
	models[model_number+4].add(layers.MaxPooling2D((2,2)))
	models[model_number+4].add(layers.Conv2D(64, (3, 3), activation='relu'))
	models[model_number+4].add(layers.Flatten())
	models[model_number+4].add(layers.Dense(64, activation = 'relu'))
	models[model_number+4].add(layers.Dense(2))
	models[model_number+4].compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), metrics = ["accuracy"])
	implementations[model_number+4] = models[model_number+4].fit(training_img, training_prediction_labels, epochs = 2, validation_data=(testing_img, validation_prediction_labels), verbose = 2)


#Model 1 + Predictor
create_model(0, 0)
create_predictor(0)

#Model 2 + Predictor
create_model(1, 0.2)
create_predictor(1)

#Model 3 + Predictor
create_model(2, 0.4)
create_predictor(2)

#Model 4 + Predictor
create_model(3, 0.6)
create_predictor(3)

i = 0
for model in models:
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model.
  filename = "model" + str(i) + ".tflite"
  with open(filename, 'wb') as f:
    f.write(tflite_model)
  i++
