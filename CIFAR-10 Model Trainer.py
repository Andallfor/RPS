#tar.gz is monkEEEEEEEE :() :() water monkE moment 
import tensorflow as tf
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import BatchNormalization
import ssl
import sys
import datetime

#directory where it saves model
checkpoint_path = "training_1/cp.ckpt"

#Get the dataset ready monke :() :()
ssl._create_default_https_context = ssl._create_unverified_context
(training_img, training_label), (testing_img, testing_label) = tf.keras.datasets.cifar10.load_data() #fine_grained labels, classes not superclasses
training_img = training_img / 255 #normalizing pixel values https://m.youtube.com/watch?v=T78nq62aQgM
testing_img = testing_img / 255
number_of_classes = 10

#https://keras.io/api/layers/preprocessing_layers/image_augmentation/
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2)
  #layers.RandomRotation(0.2)
])

#Creates the model
model = tf.keras.Sequential()

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

model.add(BatchNormalization())
model.add(Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#sets up tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#trains the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(training_img, training_label, epochs = 1, validation_data=(testing_img, testing_label), callbacks=[tensorboard_callback]) #increase epoch length later

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
