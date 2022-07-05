import os
import re
from tkinter.tix import Tree
import numpy as np
import cv2
import json
from img_utils import savefig_keyboard,savefig_keyboard_hand
from captureFigures import capture_figures
# from rgb2gray import rgb2gray
from RotateFigures import rotate_figure
from cropFigure import crop_fig,crop_fig_bkr
from mostBrightness import brightness
from key_detection import key_detection, get_keys_visual
from backgroundFilter import moveTowards_filter
from get_key import get_key
from drawRect import hand_crop
def get_background(current_pwd):
    overwrite = True
    readfile = False
    
    max_brighness=0.0
    bkg_img = None
    bgr_path = None
    bgr_mask = None
    count = 0
    if not readfile:
        # for path in sorted(os.listdir(current_pwd + "testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())):            
        #     # fig_gray = rgb2gray(path)
        #     figurePath = os.path.join(current_pwd + "testFigures",path)
        #     figure = cv2.imread(figurePath)
        #     # rotate_figure(fig_gray)

        #     fig_keyboard,whitekey,mask = crop_fig(figure, path)
        #     if len(fig_keyboard)==0 or len(whitekey)==0:
        #        continue
        bgr_mask = hand_crop(current_pwd)
            


        # print(bgr_path,max_brighness)
        
        for path in sorted(os.listdir(current_pwd + "testFigures"),key = lambda i:int(re.match(r'(\d+)',i).group())): 
            figurePath = os.path.join(current_pwd + "testFigures",path)
            figure = cv2.imread(figurePath)
            keyboard = crop_fig_bkr(figure, bgr_mask)
            if keyboard.shape[0]==0 or keyboard.shape[1]==0:
                continue
            savefig_keyboard_hand(current_pwd, path,keyboard,overwrite)

            b = brightness(keyboard)
            print(b,path)

            if b > max_brighness:
               max_brighness = b
               bkg_img = keyboard
               bgr_path = path
        with open(current_pwd + 'data.json', 'w') as f:
            data={}
            data["bgr_path"]=bgr_path
            json.dump(data,f)
    else:
        with open(current_pwd + 'data.json') as f:
            data = json.load(f)
        # print("read {}".format(path))
        bgr_path = data["bgr_path"]
        bkg_img = cv2.imread(os.path.join(current_pwd + "testFigures_keyboard",bgr_path))
    return bkg_img

def keyboard_segmentation(bkg_img):
    keyboard, black_keys, white_keys = key_detection(bkg_img)
    return keyboard, black_keys, white_keys

def get_keys(bgr,keyboard, black_keys, white_keys):
    for path in sorted(os.listdir("./testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
        frame = cv2.imread(os.path.join("./testFigures_keyboard",path) , cv2.IMREAD_COLOR)
        # import pdb;pdb.set_trace()
        frm = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(bgr[0])!=len(frm[0]):
            continue
        res_black = moveTowards_filter(bgr,frm, 30)
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
        result_img_visual = get_keys_visual(frame, black_keys, white_keys, keys)
        cv2.imshow("result_img_visual", result_img_visual )
        cv2.waitKey(0)


if __name__ == '__main__':
    video_path = r"\\EVAN\pianoyt_video\video_100.mp4"
    capture_figures(video_path,"./testFigures")
    bgr = get_background()
    # cv2.imshow("result_img", bgr)
    # cv2.waitKey(0)
    keyboard, black_keys, white_keys = keyboard_segmentation(bgr)
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    get_keys(bgr,keyboard,black_keys, white_keys)
    # pdb.set_trace()

