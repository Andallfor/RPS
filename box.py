import numpy as np

class box:
    def __init__(self, data, value, size, position, pixels):
        self.data = data
        self.value = value
        self.position = position
        self.size = size
        self.pixels = pixels
        
        # combine pixels
        ll = point(pixels[0].position.x, pixels[0].position.y)
        ur = point(pixels[0].position.x, pixels[0].position.y)
        for p in self.pixels: # get boundary of pixels
            if p.position.x < ll.x:
                ll.x = p.position.x
            if p.position.y < ll.y:
                ll.y = p.position.y
            if p.position.x + p.outline.shape[1] > ur.x:
                ur.x = p.position.x + p.outline.shape[1]
            if p.position.y + p.outline.shape[0] > ur.y:
                ur.y = p.position.y + p.outline.shape[0]

        self.combinedPixels = np.full((ur.y - ll.y, ur.x - ll.x), False, dtype=bool)
        validValues = [p.key for p in self.pixels]
        
        imgMask = self.pixels[0].fullMask
        for y in range(ll.y, self.combinedPixels.shape[0] + ll.y):
            for x in range(ll.x, self.combinedPixels.shape[1] + ll.x):
                if imgMask[y, x] in validValues:
                    self.combinedPixels[y - ll.y, x - ll.x] = True
                else:
                    self.combinedPixels[y - ll.y, x - ll.x] = False
        
        # find perimeter
        #perm = self.checkChunk(self.combinedPixels, False, point(0, 0), point(self.combinedPixels.shape[0], self.combinedPixels.shape[1]), 0, 1)
    
    # assumes shape is square
    def checkChunk(self, values, important, minPoint, maxPoint, step, limit):
        # is completely full
        if np.all(values):
            if important:
                return [self._avg(minPoint, maxPoint)]
        
        if np.all(values == False):
            return [-1]

        if step > limit:
            return [self._avg(minPoint, maxPoint)]
        
        l = int(values.shape[0] / 2)
        w = point(maxPoint.x / 2, maxPoint.y / 2)
        v = []
        v += self.checkChunk(np.array(values[0:l, 0:l]), True, minPoint, w, step + 1, limit)
        v += self.checkChunk(np.array(values[l:l+l, 0:l]), True, point(minPoint.x + w.x, minPoint.y), point(w.x + w.x, w.y), step + 1, limit)
        v += self.checkChunk(np.array(values[0:l, l:l+l]), True, point(minPoint.x, minPoint.y + w.y), point(w.x, w.y + w.y), step + 1, limit)
        v += self.checkChunk(np.array(values[l:l+l, l:l+l]), True, w, maxPoint, step + 1, limit)
        
        v = [_v for _v in v if _v != -1]
        return v
    
    def _avg(self, minPoint, maxPoint):
        return point((maxPoint.x + minPoint.x) / 2, ((maxPoint.y + minPoint.y) / 2))
                    
class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"


class superPixel:
    def __init__(self, key, position, outline, fullMask):
        self.key = key
        self.position = position
        self.outline = outline
        self.fullMask = fullMask
    
    def withinBox(self, minX, minY, maxX, maxY):
        return self.position.x > minX and\
               self.position.y > minY and\
               self.position.x + self.outline.shape[0] < maxX and\
               self.position.y + self.outline.shape[1] < maxY
    
    def containsPoint(self, point):
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
