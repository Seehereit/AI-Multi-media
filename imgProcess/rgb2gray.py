import os
import re
import cv2 as cv
import pdb

def rgb2gray(path):
    figurePath = os.path.join("./testFigures",path)
    image = cv.imread(figurePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray image',gray)
    # cv.imwrite(os.path.join(save_path,p),gray)
    return gray
    # cv.waitKey(0)
    # cv.destroyAllWindows() 