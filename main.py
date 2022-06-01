import os
import re
import numpy as np
import cv2
import json
from utils import savefig_keyboard
from captureFigures import capture_figures
# from rgb2gray import rgb2gray
from RotateFigures import rotate_figure
from cropFigure import crop_fig,crop_fig_bkr
from mostBrightness import brightness
from key_detection import key_detection, get_keys_visual
from backgroundFilter import moveTowards_filter
from get_key import get_key

def get_background():
    overwrite = False
    readfile = True
    
    max_brighness=0.0
    bkg_img = None
    bgr_path = None
    bgr_mask = None
    count = 0
    if not readfile:
        for path in sorted(os.listdir("./testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())):            
            # fig_gray = rgb2gray(path)
            figurePath = os.path.join("./testFigures",path)
            figure = cv2.imread(figurePath)
            # rotate_figure(fig_gray)
            # import pdb;pdb.set_trace()
            fig_keyboard,mask = crop_fig(figure)
            if len(fig_keyboard)==0:
               continue
           
            # import pdb;pdb.set_trace()
            count = count + 1
            b = brightness(fig_keyboard)
            # cv2.imshow("result_img",whitekey)
            # cv2.waitKey(0)
            print(b,path)
            if b > max_brighness:
               max_brighness = b
               bkg_img = fig_keyboard
               bgr_path = path
               bgr_mask = mask
            if count>300:
                break


        print(bgr_path,max_brighness)
        
        for path in sorted(os.listdir("./testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())): 
            figurePath = os.path.join("./testFigures",path)
            figure = cv2.imread(figurePath)
            keyboard = crop_fig_bkr(figure, bgr_mask)
            if keyboard.shape[0]==0 or keyboard.shape[1]==0:
                continue
            savefig_keyboard(path,keyboard,overwrite)
        with open('./data.json', 'w') as f:
            data={}
            data["bgr_path"]=bgr_path
            json.dump(data,f)
    else:
        with open('./data.json') as f:
            data = json.load(f)
        # print("read {}".format(path))
        bgr_path = data["bgr_path"]
        bkg_img = cv2.imread(os.path.join("./testFigures_keyboard",bgr_path))
    return bkg_img

def keyboard_segmentation(bkg_img):
    keyboard, black_keys, white_keys = key_detection(bkg_img)
    return keyboard, black_keys, white_keys

def get_keys(bgr,keyboard):
    for path in sorted(os.listdir("./testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
        frm = cv2.imread(os.path.join("./testFigures_keyboard",path) , cv2.IMREAD_GRAYSCALE)
        # import pdb;pdb.set_trace()
        # frm = np.array(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
        if len(bgr[0])!=len(frm[0]):
            continue
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
        # cv2.imshow("result_img", thresh_black )
        # cv2.waitKey(0)


if __name__ == '__main__':
    video_path = r"\\EVAN\pianoyt_video\video_100.mp4"
    #capture_figures(video_path,"./testFigures")
    bgr = get_background()
    # cv2.imshow("result_img", bgr)
    # cv2.waitKey(0)
    keyboard, black_keys, white_keys = keyboard_segmentation(bgr)
    result_img = get_keys_visual(bgr, black_keys, white_keys, [29, 62, 71])
    cv2.imshow("result_img", result_img)
    cv2.waitKey(0)
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    #get_keys(bgr,keyboard)
    # pdb.set_trace()

