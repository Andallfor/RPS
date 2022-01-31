from matplotlib.font_manager import json_dump
import numpy as np
import cv2
import skimage.measure
from box import box, point
import objectDetection
import SNMS
from typing import Tuple, List
import time
import json
import math
from skimage.segmentation import mark_boundaries, felzenszwalb as flz
import matplotlib.pyplot as plt
import os

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class main:
    def __init__(self, poolSize: int, sizes: List[int], threshold: float):
        self.width: int = 1280
        self.height: int = 720
        self.poolSize: int = poolSize
        self.sizes: List[int] = sizes
        self.threshold: float = threshold
    
    def start(self):
        self.v = cv2.VideoCapture(0)
        self.v.set(3, self.width)
        self.v.set(4, self.height)
    
    def update(self, index, image=True):
        baseI = self.v.read()[1]
        poolI = skimage.measure.block_reduce(baseI, (self.poolSize, self.poolSize, 1), np.max)
        pixels = objectDetection.superPixels(poolI, 5, 100, 3, 25, index)
        
        boxes = objectDetection.drawBoxes(poolI, self.sizes, pixels)
        if len(boxes) == 0:
            # no solution
            return skimage.measure.block_reduce(baseI, (2, 2, 1), np.max)

        solution = SNMS.method2(boxes, self.threshold, count=3)
        if image:
            #baseI = mark_boundaries(baseI, flz(baseI, 150, 3, 100))
            lineI = SNMS.drawOutline(solution, self.poolSize, skimage.measure.block_reduce(baseI, (2, 2, 1), np.max))
            data = self.findDist(solution)
            for d in data:
                lineI = cv2.putText(lineI, 
                                    str(d[1]) + '|' + str(d[2]),
                                    (int(d[0].x * self.poolSize / 2), int(d[0].y * self.poolSize / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1,
                                    (255, 0, 0))
            # draw equation line
            n = 5
            d = 8
            for i in range(n + 1):
                x = int(lineI.shape[1] / 2)
                y = lineI.shape[0] - int((lineI.shape[0] / 2) * (i / n))
                lineI = cv2.putText(lineI, f'- {i / n}', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
            return lineI
        else:
            return solution
    
    def saveFrame(self, boxes):
        name: str = str(int(time.time())) + '.json'
        
        f = open(name, 'w')
        data = dict()
        data['length'] = len(boxes)
        
        _d = dict()
        for i, _b in enumerate(boxes):
            b = [_b.position.x, _b.position.y, _b.size, _b.data]
            d = b[3].flatten()
            d = d.astype(str)
            _d[i] = {'x': int(b[0]), 'y': int(b[1]), 'size': int(b[2]), 'data': [i for i in d]}
        
        data[int(time.time())] = _d
            
        f.write(json.dumps(data))
    
    def findDist(self, boxes):
        _a = 0.0133414
        _b = 32.3227
        _c = 1.68011
        _d = 18.6921
        
        kL = point(0, 0)
        kR = point(1, 0)
        kS = point(self.width / 2, self.height / 2)
        mP = (kR.y - kL.y) / (kR.x - kL.y)
        bP = kL.y - mP * kL.x
        if mP == 0:
           mP = 0.001
        mK = -1 / mP
        bK = kS.y - mK * kS.x
        p = point(-bK / mK, 0)

        screenDist = math.sqrt((p.y - kS.y) * (p.y - kS.y) + (p.x - kS.x) * (p.x - kS.x))
        slope = math.atan2(kS.y - p.y, kS.x - p.x) - math.radians(90)
        trueScreenDist = screenDist * math.cos(slope)
        
        data = []
        for b in boxes:
            uS = point(b.position.x + b.size / 2, self.height - (b.position.y + b.size) * self.poolSize) # invert y value
            uB = uS.y - mP * uS.x
            x = (uB - bK) / (mK - mP)
            y = mP * x + uB

            percent = y / trueScreenDist
            worldPercent = _a * math.pow(_b, percent + _c) + _d # unlike old code, this models the distance
            #worldPosition = round(worldPercent * knownDist, 2)
            
            data.append((point(b.position.x, b.position.y), round(worldPercent), round(self.height - (b.position.y + b.size) * self.poolSize)))
        
        return data

    def standardize(self, boxes):
        return [(b.position.x / (b.size / 32), b.position.y / (b.size / 32), cv2.resize(b.data, (32, 32))) for b in boxes]

m = main(5, [16, 32, 48, 64], 1000000000000000)
m.start()
m.update(0, image=False)
os.system('cls' if os.name == 'nt' else 'clear')
m.update(0, image=False)
os.system('cls' if os.name == 'nt' else 'clear')
print("started")
index = 0
while (True):
    img = m.update(index, image=True)
    
    index += 1
    cv2.imshow('', img)
    cv2.waitKey(0)
#data = m.update(image=True)
#m.saveFrame(data)