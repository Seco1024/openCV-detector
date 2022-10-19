import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import cv2 as cv

class TextReader:
    
    def __init__(self):
        self.__reader = easyocr.Reader(['ch_tra'])

    def toText(self, screenshot):
        blur_img = cv.GaussianBlur(screenshot, (0, 0), 100)
        sharpen = cv.addWeighted(screenshot, 1.5, blur_img, -0.5, 0)
        res = self.__reader.readtext(sharpen)
        return res
    