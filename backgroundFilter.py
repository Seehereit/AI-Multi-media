import pdb
import cv2
import numpy as np
from PIL import Image,ImageStat

def moveTowards_filter(bgr,frm,step=0):
    # pdb.set_trace()
    bgr = np.array(bgr,dtype=np.int16)
    frm = np.array(frm,dtype=np.int16)
    if bgr.shape[0]>frm.shape[0]:
        frm = np.row_stack((frm,np.zeros((bgr.shape[0]-frm.shape[0],bgr.shape[1]))))
    elif bgr.shape[0]<frm.shape[0]:
        frm = frm[0:bgr.shape[0],:]
    # pdb.set_trace()
    res=np.clip(frm + np.clip(abs(bgr-frm),a_min=0,a_max=step) * np.sign(bgr-frm),a_min=0,a_max=255)
    res = res.astype(np.uint8)
    # pdb.set_trace()
    return res

if __name__ == '__main__':
    background = "testFigures/219.bmp"
    frame = "testFigures/303.bmp"
    from cropFigure import crop_fig
    bgr = cv2.imread(background, cv2.IMREAD_COLOR)
    bgr,_ = crop_fig(bgr)
    frm = cv2.imread(frame, cv2.IMREAD_COLOR)
    frm,_ = crop_fig(frm)
    res = moveTowards_filter(bgr,frm)
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    

    white = np.clip(bgr.astype(np.int16)-res.astype(np.int16),a_min=0,a_max=255).astype(np.uint8)
    black = np.clip(res.astype(np.int16)-bgr.astype(np.int16),a_min=0,a_max=255).astype(np.uint8)
    _, thresh_white = cv2.threshold(white,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_black = cv2.threshold(black,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(src=black, thresh=30, maxval=255, type=0)
    cv2.imshow("result_img", thresh_white )
    cv2.waitKey(0)
    cv2.imshow("result_img", thresh_black )
    cv2.waitKey(0)
    cv2.destroyAllWindows()