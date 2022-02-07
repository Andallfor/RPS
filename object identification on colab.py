from keras.layers.preprocessing.image_preprocessing import RandomZoom
from keras import layers
from keras.layers import RandomZoom
#yeah i'm not cleaning up the imporrts
from matplotlib.cbook import flatten
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import RandomTranslation
from keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

cifar=tf.keras.datasets.cifar10
(tran_images,tran_labels),(tet_images,tet_labels)=cifar.load_data()


tran_images=tran_images/255.0
tet_images=tet_images/255.0

print(tran_images.shape)
augmentation=tf.keras.Sequential([RandomFlip(mode="horizontal"),RandomZoom(height_factor=(-.3,.3))])#augmentation layer. I just zoom and flip horizontally. Any suggestions on what else to do?
modell=Sequential()
modell.add(augmentation)                           
modell.add(Conv2D(32,(3,3),activation='relu',padding='same'))
modell.add(BatchNormalization())
modell.add(Conv2D(32,(3,3),activation='relu',padding='same'))

modell.add(BatchNormalization())
modell.add(MaxPooling2D((2,2)))
modell.add(Dropout(0))
modell.add(Conv2D(64,(3,3),activation='relu',padding='same'))
modell.add(BatchNormalization())
modell.add(Conv2D(64,(3,3),activation='relu',padding='same'))

modell.add(BatchNormalization())
modell.add(MaxPooling2D((2,2)))
modell.add(Dropout(0))
modell.add(Conv2D(128,(3,3),activation='relu',padding='same'))
modell.add(BatchNormalization())
modell.add(Conv2D(128,(3,3),activation='relu',padding='same'))

modell.add(BatchNormalization())
modell.add(Dropout(0))
modell.add(Flatten())
modell.add(Dropout(0.2))
modell.add(Dense(256,activation='relu'))
modell.add(Dropout(0.2))
                         
modell.add(Dense(10,activation='softmax'))
modell.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
#print(cifclassnems[history[0]])
#print('the rock')
print(tran_images.shape)
print(tran_labels)
print(tran_images)
modell.fit(tran_images,tran_labels,epochs=10, validation_data=(tet_images, tet_labels))
!mkdir -p saved_model

print(tran_labels[0])

#--------------------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')
#--------------------------------------------------------------------------------------
#predictions and image inports
#you have to have a folder in your main drive called test and inside of that have a folder called images and that folder is filled with images
from tensorflow.python.framework.dtypes import as_dtype
import skimage as sk
import cv2 as cv
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import os
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pl

from keras.preprocessing.image import DirectoryIterator
from google.colab import files
paths=[]
imgs=[]
cifclassnems=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
d='/content/drive/MyDrive/test/images'
#so what this code does is it takes the path of the images folder, then it cycles through every image and adds the image paths to the paths array. Then we use cv.imread to read all the files
for path in os.listdir(d):
  paths.append(os.path.join(d,path))
for i in paths:
  h=cv.cvtColor(cv.imread(i),cv.COLOR_BGR2RGB)
  #h=sk.img_as_float32(h)
  
  imgs.append(h)
#imgs=np.array(imgs)
for i in range(len(imgs)):
  imgs[i]=cv.resize(imgs[i],(32,32))
finim=tf.convert_to_tensor(imgs) #convert image arrays to tensors

finim=finim/255#normalize pixel values
finim=tf.random.shuffle(finim)#shuffle the first dimension of the tensor so that we get random images
#print(test_set[0])


#imarray=np.array([img_to_array(img)])

!ls saved_model
#modelll=tf.keras.models.load_model('/content/saved_model/mymode3')
modelll=modell#load model from before. the line above kinda broke so i just reference it

#print(tet_images[0])
history=modelll(finim)


print(np.argmax(history[0]))
probability_model1 = tf.keras.Sequential([modelll, 
                                         tf.keras.layers.Softmax()])
prediction1=probability_model1.predict(finim)
prediction2=modelll.predict(finim)#THIS IS THE MAIN PREDICTION FUNCTION. THe others do the same thing but this gave good confidence values
print(prediction2[0])
#print('10')

history1=np.argmax(prediction1[0])
#print(test_set.__getitem__(0))
print(history1)
#print(history1)
pl.figure(figsize=(40,40))
for l in range(25):#go through the images and display them using pyplot
  hist=np.argmax(prediction2[l])
  pl.subplot(5,5,l+1)
  pl.imshow(finim[l])
  pl.ylabel(cifclassnems[hist]+str(prediction2[l][hist])+cifclassnems[np.argmax(prediction1[l])])


  


