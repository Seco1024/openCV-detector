import cv2 as cv
import numpy as np

class Vision:
    template = None
    template_w = 0
    template_h = 0

    def __init__(self, template_path, method = cv.TM_CCOEFF_NORMED):
        
        if template_path:
            self.__template = cv.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
            self.__template_w = self.__template.shape[1]
            self.__template_h = self.__template.shape[0]
            self.__template = self.__template[:,:,:3]
            
        self.__method = method


    def templateMatching(self, img, threshold = 0.7, mode = None):

        result = cv.matchTemplate(img, self.__template, self.__method)
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        if not locations:
            return np.array([], dtype = np.int32).reshape(0, 4)

        self.__rectangles = []
        self.__points = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.__template_w, self.__template_h]
            self.__rectangles.append(rect)
            self.__rectangles.append(rect)
        self.__rectangles, weights = cv.groupRectangles(self.__rectangles, groupThreshold = 1, eps = 0.5)

        if len(self.__rectangles):
            #print('Found Matching.')

            for (x, y, w, h) in self.__rectangles:
                target_x = x + int(w / 2)
                target_y = y + int(h / 2)
                self.__points.append((target_x, target_y))
                
                
        if mode == 'rectangles':    
            return self.__rectangles
        else:
            return self.__points
    
    
    def drawCrosshairs(self, img, points):
                    
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS
        for (target_x, target_y) in points:
            cv.drawMarker(img, (target_x, target_y), color = marker_color, 
                                  markerType = marker_type, markerSize = 40, thickness = 2)

        return img


    def drawRectangles(self, img, rectangles):
        
        line_color = (255, 255, 255)
        line_type = cv.LINE_4
        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv.rectangle(img, top_left, bottom_right, color = line_color, 
                                lineType = line_type, thickness = 2)
            
        return img
