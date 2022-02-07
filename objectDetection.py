import math
import numpy as np
from box import box, superPixel, point
import cv2
from skimage.segmentation import felzenszwalb as flz, mark_boundaries
from typing import List, Any, Dict, Tuple, Type
import matplotlib.pyplot as plt
from nptyping import NDArray

image = Type[NDArray[(Any,Any,3), int]]

def pixelScore(pixel: superPixel):
    area = pixel.outline.shape[0] * pixel.outline.shape[1]
    return -0.025 * area + 1000

# comments? whats that?
def boxes(img: image, sizes: List[int], pixels: List[superPixel])->List[box]:
    threshold = 0.95
    boxes = []
    for pixel in pixels:
        valid = np.count_nonzero(pixel.outline == pixel.key)
        if valid / (pixel.outline.shape[0] * pixel.outline.shape[1]) >= threshold:
            length = min(pixel.outline.shape[0], pixel.outline.shape[1]) - 1
            
            if length == 0:
                continue
            
            insidePixels = [pixel]
            for p in pixels:
                # check for pixels inside
                if p == pixel:
                    continue
                
                if p.withinBox(pixel.position.x, pixel.position.y, pixel.position.x + length, pixel.position.y + length):
                    insidePixels.append(p)
            
            boxes.append(box(
                data=img[pixel.position.y : pixel.position.y + length, pixel.position.x : pixel.position.x + length],
                value=100,
                size=length,
                position=pixel.position,
                pixels=insidePixels
            ))
    return boxes
    
    '''
    allBoxes = []
    maxSize = 64
    distanceIncrement = 2
    for pixel in pixels:
        # get center point
        lx, ly = pixel.outline.shape
        center: point = pixel.position + point(lx, ly)
        
        boxes = []
        currentPixels: List[superPixel] = []
        currentScore = 0
        unaccountedPixels: List[superPixel] = list(pixels)
        distance = distanceIncrement
        buffer: List[superPixel] = []
        # draw boxes
        while True:
            base = center - point(distance, distance)
            length = distance * 2
            
            boundingBox = [base, point(base.x + length, base.y), point(base.x, base.y + length), point(base.x + length, base.y + length)]
            
            valid = True
            for bb in boundingBox:
                if bb.x <= 0 or bb.y <= 0 or bb.x >= img.shape[1] or bb.y >= img.shape[0]:
                    valid = False
                    break
                
                # find new pixels
                for uPixel in list(unaccountedPixels):
                    if uPixel.containsPoint(bb):
                        buffer.append(uPixel)
                        unaccountedPixels.remove(uPixel)
                        currentScore += pixelScore(uPixel)
                        break
            
            # for each superpixel in buffer, check if their fully inside our current box
            for p in list(buffer):
                if p.withinBox(base.x, base.y, base.x + length, base.y + length):
                    currentPixels.append(p)
                    buffer.remove(p)
            #currentPixels += buffer
            
            # box left image, break
            if not valid:
                break
            
            if len(currentPixels) > 0:
                boxes.append(box(
                    data=img[base.y : base.y + length, base.x : base.x + length],
                    value=math.log2(currentScore / (1.1 ** distance)),
                    size=length,
                    position=base,
                    pixels=list(currentPixels)
                ))
            
            distance += distanceIncrement
            
            # exceeded max length
            if distance >= maxSize / 2:
                break
        allBoxes += boxes
    
    return allBoxes
    '''

def superPixels(img: image, poolSize: int, scale: int, sigma: float, minSize: int):
    img = cv2.GaussianBlur(img, (poolSize, poolSize), 0.5)
    segments = flz(img, scale=scale, sigma=sigma, min_size=minSize)
    
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
