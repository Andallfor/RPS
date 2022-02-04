# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:55:13 2022

imports the model and runs predictions on it

@author: hudso
"""
import numpy as np #for arrays
import pandas as pd #data manipulation and processing
import sys
import math
import cv2 as cv
from tensorflow import keras
from joblib import load
from sklearn.impute import SimpleImputer #fill out missing data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import glob

#imoprt image
image_input = np.asarray([cv.imread(file, 1) for file in glob.glob("dataset/*.jpg")])
for i in range(0, np.size(image_input, 0)):
    image_input = cv.resize(image_input[i], (32, 32), interpolation = cv.INTER_AREA)
    image_input = np.expand_dims(image_input, 0)
image_input = image_input[:, :, :, [2, 1, 0]] #goes from weird BGR to RGB
np.save("preprocessed_data", image_input)
print(image_input.size)

#load model and deencode
ann = keras.models.load_model('training_1/cp.ckpt')
ann.summary()
predictedResults = ann.predict(image_input)

prediction = []
prediction = np.argmax(predictedResults, axis=1, out=None)
#monkE fixEd this codE for you monkE scratch monkE back monkEEEEE
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

labels = []
for i in range(np.size(prediction)):
    labels.append(class_names[prediction[i]])
