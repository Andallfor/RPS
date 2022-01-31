import numpy as np
import skimage
from skimage.segmentation import felzenszwalb as flz
import skimage.measure
import cv2
import random
import time

POOLSIZE = 5
img = cv2.imread('test3.jpg')
img = skimage.measure.block_reduce(img, (POOLSIZE, POOLSIZE, 1), np.max)
img = cv2.GaussianBlur(img, (5, 5), 0.3)

a = time.process_time()
segements = flz(img, scale=150, sigma=0.95, min_size=50)
print(len(segements))
print(time.process_time() - a)

updated = np.empty((img.shape[0], img.shape[1], 3))
colors = dict()
count = len(np.unique(segements))
i = 0
for y in segements:
    for j, x in enumerate(y):
        if x not in colors:
            colors[x] = [random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255]
        updated[i, j] = colors[x]
    i += 1

print(time.process_time() - a)
cv2.imshow('', updated)

cv2.waitKey(0)