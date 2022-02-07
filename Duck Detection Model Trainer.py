#tar.gz is monkEEEEEEEE :() :() water monkE moment 
import tensorflow as tf
import numpy as np
import ssl
import sys
import datetime
import cv2 as cv
import glob
from sklearn.model_selection import train_test_split

#directory where it saves model
checkpoint_path = "training_2/cp.ckpt"

#Get the dataset ready monke :() :()
ssl._create_default_https_context = ssl._create_unverified_context
image_input = [cv.imread(file, 1) for file in glob.glob("C:/Users/hudso/Downloads/dataset/Rubber_Ducks/*.jpg")]

#BGR to RGB
for i in range(0, np.size(image_input, 0)):
    image_input[i] = cv.cvtColor(image_input[i], cv.COLOR_BGR2RGB)

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
np.save("preprocessed_data", image_input)

#creates "labels"
labels = []
for i in range(np.size(image_input, 0)):
    labels.append(0)
    
#the not-duck dataset, literally just cifar-10
(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data()
training_img = training_img / 255
training_img = training_img[0:np.size(image_input, 0)] #takes an equal amount of cifar as ducks
for i in range(np.size(image_input, 0)): #same as prev line
    labels.append(1)
image_input = np.concatenate((image_input, training_img), axis=0)

#train/test split
img_train, img_test, label_train, label_test = train_test_split(image_input, labels, test_size = 0.25)
label_train = np.array(label_train)
label_test = np.array(label_test)

#https://keras.io/api/layers/preprocessing_layers/image_augmentation/
"""data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  #layers.RandomRotation(0.2)
])"""


#Creates the model
model = tf.keras.Sequential()
#model.add(data_augmentation)
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2,activation='softmax'))

#sets up tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #tensorboard --logdir C:/Users/hudso/Downloads/logs/fit
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#trains the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(img_train, label_train, epochs = 5, validation_data=(img_test, label_test), callbacks=[tensorboard_callback]) #increase epoch length later

#saves final version checkpoint if prompted to
savemanual = input("save trained weights?")
if savemanual == "y" or savemanual == "Y":
  model.save(checkpoint_path)
  print("lifE could bEEEE a drEam, checkpoint savEd") #courtesy of the one and only jErry
elif savemanual == "n" or savemanual == "N":
  print("monkE makE modEl go kabloomy")
  sys.exit()
else:
  print("invalid input, please type y or n")
  sys.exit()

#monkEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE