from multiprocessing import pool
import numpy as np
import math
from box import *
import random
from typing import Dict, Tuple, List, Any
from nptyping import NDArray

def iou(box1: box, box2: box)->float:
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    xA = max(box1.position.x, box2.position.x)
    yA = max(box1.position.y, box2.position.y)
    xB = min(box1.position.x + box1.size, box2.position.x + box2.size)
    yB = min(box1.position.y + box1.size, box2.position.y + box2.size)
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area1 = ((box1.position.x + box1.size) - box1.position.x + 1) * ((box1.position.y + box1.size) - box1.position.y + 1)
    area2 = ((box2.position.x + box2.size) - box2.position.x + 1) * ((box2.position.y + box2.size) - box2.position.y + 1)
    
    return intersection / float(area1 + area2 - intersection)
    
    '''
    xLap = max(0, min(box1.position.x + box1.size, box2.position.x + box2.size) - max(box1.position.x, box2.position.x))
    yLap = max(0, min(box1.position.y + box1.size, box2.position.y + box2.size) - max(box1.position.y, box2.position.y))
    intersection = xLap * yLap
    union = ((box1.size * box1.size) - intersection) + ((box2.size * box2.size) - intersection) + intersection
    return intersection / union
    '''

def snms(b: List[box], t: float, score: float = 0)->List[box]:
    b = list(b)
    
    solutions = []
    for i in range(len(b)):
        bi = max(b, key = lambda b: b.value)
        if (bi.value > score):
            solutions.append(bi)
        else:
            break
        b.remove(bi)

        # create shallow copy to prevent linking
        boxesCopy = list(b)
        for bj in boxesCopy:
            iouScore = iou(bj, bi)
            if iouScore < t:
                bj.value = bj.value * (1 - iouScore)

    return solutions

def drawBoxes(boxes: List[box], poolSize: int, img: picture)->picture:
    for b in boxes:
        x = b.position.x * poolSize
        y = b.position.y * poolSize
        length = b.size * poolSize
        
        bgr = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for a in range(length):
            img[y + a, x] = bgr
            img[y + a, x + length] = bgr
        for b in range(length):
            img[y, x + b] = bgr
            img[y + length, x + b] = bgr
    return img

def drawPixels(pixels: List[superPixel])->picture:
    # just uses the full mask lol
    fullMask: pixel2D = pixels[0].fullMask
    img = np.ones(shape=(fullMask.shape[0], fullMask.shape[1], 3), dtype=float)
    
    count = 1 / len(pixels)
    for index, x in np.ndenumerate(fullMask):
        img[index] = [count * x, count * x, count * x]

    return img