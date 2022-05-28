import os
import cv2 as cv
def savefig_keyboard(path,fig):
    dir_path = 'testFigures_keyboard'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path) 
    fig_path = os.path.join(dir_path,path)
    cv.imwrite(fig_path,fig)