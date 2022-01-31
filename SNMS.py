import numpy as np
import cv2
from box import box
import random

def iou(box1, box2):
    xLap = max(0, min(box1.position.x + box1.size, box2.position.x + box2.size) - max(box1.position.x, box2.position.x))
    yLap = max(0, min(box1.position.y + box1.size, box2.position.y + box2.size) - max(box1.position.y, box2.position.y))
    intersection = xLap * yLap
    union = ((box1.size * box1.size) - intersection) + ((box2.size * box2.size) - intersection) + intersection
    return intersection / union

def method1(b, count=5):
    lBoxes = list(b.values())
    
    solutions = []
    for i in range(count):
        best = max(lBoxes, key = lambda b: b.value)
        listPos = lBoxes.index(best)
        del lBoxes[listPos - 2 : listPos + 1]
        solutions.append(best)
    
    return solutions

def method2(b, t, count=5):
    b = dict(b)
    
    solutions = []
    for i in range(count):
        bi = max(b.values(), key = lambda b: b.value)
        solutions.append(bi)
        b.pop((bi.position.x, bi.position.y))

        # create shallow copy to prevent linking
        boxesCopy = dict(b)
        for bj in boxesCopy.values():
            iouScore = iou(bj, bi)
            if iouScore < t:
                bj.value = bj.value * (1 - iouScore)

    return solutions[0:count]

def drawOutline(boxes, poolSize, img):
    for b in boxes:
        row, column = int(b.position.y * (poolSize / 2)), int(b.position.x * (poolSize / 2))
        shape = (int(b.size * (poolSize / 2)), int(b.size * (poolSize / 2)))
        
        bgr = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for r in range(shape[0]):
            img[row + r, column] = bgr
            img[row + r, column + shape[1]] = bgr
        for c in range(shape[1]):
            img[row, column + c] = bgr
            img[row + shape[0], column + c] = bgr
    return img