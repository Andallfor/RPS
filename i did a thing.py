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
from tensorflow import keras
from joblib import load
from sklearn.impute import SimpleImputer #fill out missing data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

image_input = "lol insert something here idfk"


#load model and deencode
ann = keras.models.load_model('training_1/cp.ckpt')
ann.summary()
predictedResults = ann.predict(image)
predictedRound = []
for i in range(len(predictedResults)):
    predictedRound.append(predictedResults[i].round())
predictedRound = list(map(int, predictedRound))

#un-round
confidence = []
for i in range(len(predictedFinal)):
    lowerBound = abs(predictedResults[i] - math.floor(predictedResults[i]))
    upperBound = abs(predictedResults[i] - math.ceil(predictedResults[i]))
    if lowerBound > upperBound:
        confidence.append(upperBound)
    elif upperBound > lowerBound:
        confidence.append(lowerBound)
    elif lowerBound == upperBound:
        confidence.append(0.5)
    else:
        print("Oh no! \nThere was an issue in the confidence measuring section of the script. \nThe program will now abort")
        sys.exit()
