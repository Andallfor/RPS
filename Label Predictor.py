# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2 14:55:13 2022

Imports the model and runs predictions on it
Outputs are a matplotlib graph and a list of labels 

@author: hudso
"""

import numpy as np #for arrays
import cv2 as cv
from tensorflow import keras
import glob
import matplotlib.pyplot as plt

#imports images
image_input = [cv.imread(file, 1) for file in glob.glob("dataset/*.jpg")]

#resizes images and removes photos that failed to load
hudsonisanidiot = 0
buffer = 0 #needs to accomodate for updated indexes after removing elements
for i in range(0, np.size(image_input, 0)):
    iso_var = image_input[i - buffer]
    try:
        image_input[i - buffer] = cv.resize(iso_var, (32, 32), interpolation = cv.INTER_AREA)
    except:
        hudsonisanidiot += 1
        print(str(hudsonisanidiot) + " images failed to load get good lol")
        buffer += 1
        del image_input[i]
image_input = np.asarray(image_input)

#changes from BGR to RGB and saves data
image_input = image_input[:, :, :, [2, 1, 0]]
np.save("preprocessed_data", image_input)
image_input = image_input / 255

#loads model and deencode
ann = keras.models.load_model('training_1/cp.ckpt')
ann.summary()
predictedResults = ann.predict(image_input)

#makes predictions
prediction = []
for i in range(0, np.size(predictedResults, 0)):
    prediction = np.argmax(predictedResults, axis=1, out=None)
#monkE fixEd this codE for you monkE scratch monkE back monkEEEEE -jerry
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
labels = []
for i in range(np.size(prediction)):
    labels.append(class_names[prediction[i]])
print(labels)

#number of graphs in the visualization
if (np.size(prediction) > 25):
    graphs = 25
else:
    graphs = np.size(prediction, 0)
    
#visualizes predictions
plt.figure(figsize=(10,10))
for i in range(graphs):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_input[i])
    plt.xlabel(class_names[prediction[i]])
