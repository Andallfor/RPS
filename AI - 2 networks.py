import tensorflow as tf
from keras import layers, models, datasets
import matplotlib as plt
import numpy as np
import cv2 as opencv
from collections import Counter

(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data()

#The model
model = tf.keras.models.Sequential()
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
                           
model.add(Dense(10,activation='softmax'))
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), metrics = ["accuracy"])
implementation = model.fit(training_img, training_label, epochs = 8, validation_data=(testing_img, testing_label))

predictions = model.predict(training_img)
validation_predictions = model.predict(testing_img)
predictions = predictions.tolist()
validation_predictions = validation_predictions.tolist()
n = 0
for element in predictions:
	predictions[n] = [element.index(max(element))] #makes the array at predictions n equal the index of the largest value in the array of probablities
	n += 1

n_2 = 0
for element in validation_predictions:
	validation_predictions[n_2] = [element.index(max(element))] #same thing but with the testing images
	n_2 += 1


label_list = training_label.tolist() 
validation_label_list = testing_label.tolist()

training_prediction_labels = []
validation_prediction_labels = []

n_3 = 0
for element in predictions:
	label = label_list[n_3]
	if element[0] == label[0]:
		training_prediction_labels.append([1])#if the program guessed right, append array????
	else:
		training_prediction_labels.append([0])
	n_3 += 1

n_4 = 0
for element in validation_predictions:
	label = validation_label_list[n_4]
	if element[0] == label[0]:#if the prgram guessed right,
		validation_prediction_labels.append([1])#lol i have no idea what's happening here and why we need to append that. Aren't we going to 
	else:
		validation_prediction_labels.append([0])
	n_4 += 1


training_prediction_labels = np.array(training_prediction_labels)
validation_prediction_labels = np.array(validation_prediction_labels)


model_predictor = tf.keras.models.Sequential()
model_predictor.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
model_predictor.add(layers.MaxPooling2D((2,2)))
model_predictor.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_predictor.add(layers.MaxPooling2D((2,2)))
model_predictor.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_predictor.add(layers.Flatten())
model_predictor.add(layers.Dense(64, activation = 'relu'))
model_predictor.add(layers.Dense(2))
model_predictor.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), metrics = ["accuracy"])
implementation_2 = model_predictor.fit(training_img, training_prediction_labels, epochs = 2, validation_data=(testing_img, validation_prediction_labels))


test_predictions = []
correctness_predictions = model_predictor.predict(testing_img)
correctness_predictions = correctness_predictions.tolist()
model_predictions = model.predict(testing_img)
model_predictions = model_predictions.tolist()

n_6 = 0
for correctness in correctness_predictions:

	correct = correctness.index(max(correctness))
	
	if correct == 1:
		final_prediction = model_predictions[n_6]
		final_prediction = final_prediction.index(max(final_prediction))
		test_predictions.append(final_prediction)
	else:
		actual_prediction = model_predictions[n_6]
		actual_prediction_max = actual_prediction.index(max(actual_prediction))
		actual_prediction[actual_prediction_max] = -100
		actual_prediction = actual_prediction.index(max(actual_prediction))
		test_predictions.append(actual_prediction)

	n_6 += 1


accuracy = []
testing_label_2 = testing_label.tolist()
n_5 = 0
for prediction in test_predictions:
	
	actual_label = testing_label_2[n_5]
	actual_label = actual_label[0]

	if prediction == actual_label:
		accuracy.append(1)
	else:
		accuracy.append(0)
	
	n_5 += 1

c = Counter(accuracy)
percent_correct = c[1] / 10000
print(percent_correct)
