from PIL import Image,ImageStat
import cv2
import numpy as np  
import pdb
import os

def brightness( im_file ):
   im_pil = Image.fromarray(im_file)
   # im = Image.open(im_file).convert('L')
   im = im_pil.convert('L')
   stat = ImageStat.Stat(im)
   # pdb.set_trace()
   return stat.rms[0]

if __name__ == '__main__':
   from cropFigure import crop_fig
   # imgPath = "testFigures_gray"
   # imgs = os.listdir(imgPath)
   # max_brighness=0.0
   # img_path=None
   # for iPath in imgs:
   #    whitekey = []
   #    img = cv2.imread(os.path.join(imgPath,iPath), 0)
   #    _,whitekey = crop_fig(img)
   #    if whitekey == []:
   #       continue
   #    # cv2.imshow('whitekey', whitekey)
   #    b = brightness(img)
   #    # print(b,iPath)
   #    if b > max_brighness:
   #       max_brighness = b
   #       img_path = img
   img_path = "testFigures_gray/4969.bmp"
   from key_detection import blackkeys_detection,whitekeys_detection
   src = cv2.imread(img_path, cv2.IMREAD_COLOR)
   src,whitekey = crop_fig(src)
   black = blackkeys_detection(src)
   white = whitekeys_detection(black, np.shape(src)[0])
   src = cv2.UMat(src)
   for i in white:
          cv2.rectangle(src, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 1)
          cv2.rectangle(src, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 1)
   cv2.imshow("result_img", src)
   cv2.waitKey(0)
   # print(img_path, max_brighness)