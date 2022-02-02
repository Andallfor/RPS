import math
import numpy as np
from box import box, superPixel, point
import cv2
from skimage.segmentation import felzenszwalb as flz, mark_boundaries
from typing import List, Any, Dict, Tuple, Type
import matplotlib.pyplot as plt
from nptyping import NDArray

image = Type[NDArray[(Any,Any,3), int]]

def drawBoxes(img: image, sizes: List[int], pixels: List[superPixel])->Dict[Tuple[int,int], box]:
    boxes = dict()
    for x in range(0, 256, 4):
        for y in range(0, 144, 4):
            for size in sizes:
                if x + size >= 256 or y + size >= 144:
                    continue
                    
                b = img[y : y + size, x : x + size]
                
                validPixels = []
                score = 0
                for p in pixels:
                    if p.withinBox(x, y, x + size, y + size):
                        score += 1
                        validPixels.append(p)
                score = score / math.pow(size, 0.75)
                if score == 0:
                    continue
                
                boxes[(x, y)] = box(b, score, size, point(x, y), validPixels)
    return boxes

def superPixels(img: image, poolSize: int, scale: int, sigma: float, minSize: int, index):
    img = cv2.GaussianBlur(img, (poolSize, poolSize), 0.5)
    segments = flz(img, scale=scale, sigma=sigma, min_size=minSize)
    
    if (index == 6):
        fig = plt.figure(figsize=(12, 9))
        plt.imshow(mark_boundaries(img[:,:,::-1], segments), )
        plt.show()
    
    return segementToPixels(segments)

def segementToPixels(segements: NDArray[(Any,Any), int])->List[superPixel]:
    pixels = []
    keys = dict()
    for y, segement in enumerate(segements):
        for x, value in enumerate(segement):
            boundary = [x, y, x, y] # minX, minY, maxX, maxY
            if value in keys.keys():
                boundary = keys[value]
            
            # set bounds
            if x > boundary[2]:
                boundary[2] = x
            if x < boundary[0]:
                boundary[0] = x
            if y > boundary[3]:
                boundary[3] = y
            if y < boundary[1]:
                boundary[1] = y
            
            keys[value] = boundary

    for key, value in keys.items():
        outline = np.array(segements[value[1]:value[3] + 1, value[0]:value[2] + 1])
        outline[outline != key] = -1
        pixels.append(superPixel(key, point(value[0], value[1]), outline, segements))
            
    return pixels
