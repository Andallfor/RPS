import numpy as np
import cv2
from box import box
import random
from typing import Dict, Tuple, List, Any
from nptyping import NDArray

def iou(box1: box, box2: box)->float:
    xLap = max(0, min(box1.position.x + box1.size, box2.position.x + box2.size) - max(box1.position.x, box2.position.x))
    yLap = max(0, min(box1.position.y + box1.size, box2.position.y + box2.size) - max(box1.position.y, box2.position.y))
    intersection = xLap * yLap
    union = ((box1.size * box1.size) - intersection) + ((box2.size * box2.size) - intersection) + intersection
    return intersection / union

# TODO: replace with returning all boxes above a certain threshold
def snms(b: Dict[Tuple[int, int], box], t: float, count: int=5)->List[box]:
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

def drawOutline(boxes: List[box], poolSize: int, img: NDArray[(Any,Any,3), int])->NDArray[(Any,Any,3), int]:
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