import os, cv2, re
from glob import glob
import numpy as np
from captureFigures import capture_figures
from main import get_background
from backgroundFilter import moveTowards_filter
import json

image_paths = glob(os.path.join('mixModel/data/SIGHT', 'image', 'video_*'))
#image 文件夹

for image_path in image_paths:
    if not os.path.exists(image_path):      
        os.makedirs(image_path)
    current_pwd = image_path + "\\"
    video = image_path.replace('\\image\\', '\\video\\') + '.mp4'
    
    if not os.path.exists(current_pwd + 'testFigures_keyboard'):     #生成键盘图片集
        capture_figures(video, current_pwd + 'testFigures')
    #后面的部分放到load函数里
        bgr = get_background(current_pwd)
        testfigures = glob(os.path.join(current_pwd, 'testFigures', '*.bmp'))
        for item in testfigures:
            os.remove(item)
        os.removedirs(current_pwd + 'testFigures')
    with open(current_pwd + 'data.json') as f:
        data = json.load(f)
    # print("read {}".format(path))
    bgr_path = data["bgr_path"]
    bgr = cv2.imread(os.path.join(current_pwd + "testFigures_keyboard",bgr_path))
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if not os.path.exists(current_pwd + 'black'):
        for path in sorted(os.listdir(current_pwd + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
            frame = cv2.imread(os.path.join(current_pwd + "testFigures_keyboard",path) , cv2.IMREAD_COLOR)
            # import pdb;pdb.set_trace()
            frm = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(bgr[0])!=len(frm[0]):
                continue
            res_black = moveTowards_filter(bgr,frm, 30)
            res_white = moveTowards_filter(bgr,frm, 50)
            white = np.clip(bgr.astype(np.int16)-res_white.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
            white = cv2.resize(white, (640, 64))
            os.makedirs(current_pwd + 'white', exist_ok=True)
            cv2.imwrite(current_pwd + 'white\\' + path, white)
            black = np.clip(res_black.astype(np.int16)-bgr.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
            black = cv2.resize(black, (640, 64))
            os.makedirs(current_pwd + 'black', exist_ok=True)
            cv2.imwrite(current_pwd + 'black\\' + path, black)
    if not os.path.exists(current_pwd + 'mix'):  
        for path in sorted(os.listdir(current_pwd + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
            white = cv2.imread(os.path.join(current_pwd + "white",path), cv2.COLOR_BGR2GRAY)
            white = cv2.resize(white, (640, 64))
            black = cv2.imread(os.path.join(current_pwd + "black",path), cv2.COLOR_BGR2GRAY)
            black = cv2.resize(black, (640, 64))
            mix = np.concatenate((black, white), axis=0)
            os.makedirs(current_pwd + 'mix', exist_ok=True)
            cv2.imwrite(current_pwd + 'mix\\' + path, mix) 