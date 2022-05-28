import os
import re
import numpy as np
import cv2
from utils import savefig_keyboard
from captureFigures import capture_figures
# from rgb2gray import rgb2gray
from RotateFigures import rotate_figure
from cropFigure import crop_fig
from mostBrightness import brightness
from key_detection import blackkeys_detection,whitekeys_detection,key_detection,key_detection_visual
from backgroundFilter import moveTowards_filter
from get_key import get_key
import pdb

def get_background():
    max_brighness=0.0
    bkg_img = None
    p = None
    for path in sorted(os.listdir("./testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())):            
        # fig_gray = rgb2gray(path)
        figurePath = os.path.join("./testFigures",path)
        fig_gray = cv2.imread(figurePath)
        # rotate_figure(fig_gray)
        # import pdb;pdb.set_trace()
        fig_keyboard,whitekey = crop_fig(fig_gray)
        if len(whitekey) == 0:
           continue
        #savefig_keyboard(path,fig_keyboard)
        # import pdb;pdb.set_trace()

        b = brightness(whitekey)
        # cv2.imshow("result_img",whitekey)
        # cv2.waitKey(0)
        # print(b,iPath)
        if b > max_brighness:
           max_brighness = b
           bkg_img = fig_keyboard
           p = path
    print(p,max_brighness)

    return bkg_img

def keyboard_segmentation(bkg_img):
    keyboard = key_detection(bkg_img)
    return keyboard

def get_keys(bgr,keyboard):
    for path in sorted(os.listdir("./testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
        frm = cv2.imread(os.path.join("./testFigures_keyboard",path) , cv2.IMREAD_COLOR)
        
        frm = np.array(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
        
        res_black = moveTowards_filter(bgr,frm, 12)
        res_white = moveTowards_filter(bgr,frm, 50)
        white = np.clip(bgr.astype(np.int16)-res_white.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
        black = np.clip(res_black.astype(np.int16)-bgr.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
        # pdb.set_trace()
        #_, thresh_white = cv2.threshold(white,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #_, thresh_black = cv2.threshold(black,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        keys = []
        for i in range(1, 89):
            mask = (keyboard==i)
            if i<=52:
                target = white
            else:
                target = black
            if get_key(target,mask, i):
                keys.append(i)
            # if target[mask].sum()/255 > int(mask.sum()*2/3):
                # if 
                # keys.append(i)
        print(str(path) + str(keys))
        # pdb.set_trace()
        # cv2.imshow("result_img", thresh_black )
        # cv2.waitKey(0)
 
if __name__ == '__main__':
    video_path = r"\\EVAN\pianoyt_video\video_100.mp4"
    #capture_figures(video_path,"./testFigures")
    bgr = get_background()
    # cv2.imshow("result_img", bgr)
    # cv2.waitKey(0)
    keyboard = keyboard_segmentation(bgr)
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    get_keys(bgr,keyboard)
    # pdb.set_trace()

