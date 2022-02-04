import numpy as np
from nptyping import NDArray
import cv2
from typing import Any, List, Type

pixel = Type[NDArray[(Any,Any,3), int]]
pixel2D = Type[NDArray[(Any,Any), int]]
picture = Type[NDArray[(Any,Any,3), int]]

class point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

class box:
    def __init__(self, data: pixel, value: float, size: int, position: point, pixels: List[pixel]):
        self.data: pixel = data
        self.value: float = value
        self.position: point = position
        self.size: int = size
        self.pixels: List[pixel] = pixels
        
        # combine pixels
        ll: point = point(pixels[0].position.x, pixels[0].position.y)
        ur: point = point(pixels[0].position.x, pixels[0].position.y)
        for p in self.pixels: # get boundary of pixels
            if p.position.x < ll.x:
                ll.x = p.position.x
            if p.position.y < ll.y:
                ll.y = p.position.y
            if p.position.x + p.outline.shape[1] > ur.x:
                ur.x = p.position.x + p.outline.shape[1]
            if p.position.y + p.outline.shape[0] > ur.y:
                ur.y = p.position.y + p.outline.shape[0]

        sizeY: int = ur.y - ll.y
        sizeX: int = ur.x - ll.x
        self.combinedPixels: NDArray[(sizeY, sizeX), bool] = np.full((sizeY, sizeX), False, dtype=bool)
        validValues = [p.key for p in self.pixels]
        
        imgMask = self.pixels[0].fullMask
        for y in range(ll.y, self.combinedPixels.shape[0] + ll.y):
            for x in range(ll.x, self.combinedPixels.shape[1] + ll.x):
                if imgMask[y, x] in validValues:
                    self.combinedPixels[y - ll.y, x - ll.x] = True
                else:
                    self.combinedPixels[y - ll.y, x - ll.x] = False
    
    def _avg(self, minPoint: point, maxPoint: point):
        return point((maxPoint.x + minPoint.x) / 2, ((maxPoint.y + minPoint.y) / 2))

class frame:
    def __init__(self, base: picture, pool: picture, pixel: picture, boxes: picture, estimate: picture, solution: List[box]):
        self.base: picture = base
        self.pool: picture = pool
        self.pixel: picture = pixel
        self.boxes: picture = boxes
        self.estimate: picture = estimate
        self.solution: List[box] = solution
    
    def displayAll(self):
        #cv2.imshow('base', self.base)
        #cv2.imshow('pooled', self.pool)
        cv2.imshow('pixels', self.pixel)
        #cv2.imshow('boxes', self.boxes)
        cv2.imshow('estimated', self.estimate)
        cv2.waitKey(0)

class superPixel:
    def __init__(self, key: int, position: point, outline: pixel2D, fullMask: pixel2D):
        self.key: int = key
        self.position: point = position
        self.outline: pixel2D = outline
        self.fullMask: pixel2D = fullMask
    
    def withinBox(self, minX: point, minY: point, maxX: point, maxY: point)->bool:
        return self.position.x > minX and\
               self.position.y > minY and\
               self.position.x + self.outline.shape[0] < maxX and\
               self.position.y + self.outline.shape[1] < maxY
    
    def containsPoint(self, point: point)->bool:
        return point.x > self.position.x and\
               point.y > self.position.y and\
               point.x < self.position.x + self.outline.shape[0] and\
               point.y < self.position.y + self.outline.shape[1]

if __name__ == '__main__':
    from objectDetection import segementToPixels
    
    #fullMask = np.array([[0, 0, 0, 0, 0],
    #                     [0, 0, 1, 1, 2],
    #                     [3, 3, 1, 2, 2],
    #                     [3, 3, 3, 1, 2],
    #                     [4, 4, 4, 4, 2]])
    fullMask = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])
    
    fm = []
    height = 0
    radius = 128
    for i in range(1, int(radius / 2)):
        first = [0 for j in range(int(radius / 2) - i)]
        middle = [1 for j in range(i)]
        
        fm.append(first + middle + middle + first)
    
    for i in range(int(radius / 2), 0, -1):
        first = [0 for j in range(int(radius / 2) - i)]
        middle = [1 for j in range(i)]
        
        fm.append(first + middle + middle + first)
        
    fm = np.array(fm)
    pixels = segementToPixels(fm)
    #for p in pixels:
    #    print(p.outline, p.position, p.key)
    b = box(0, 0, (5, 5), point(0, 0), [pixels[1]])
