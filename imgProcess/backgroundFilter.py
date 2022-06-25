import pdb
import cv2
import numpy as np
from PIL import Image,ImageStat
from key_detection import key_detection_visual

def moveTowards_filter(bgr,frm,step=20):
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
    frame = "testFigures/268.bmp"
    from cropFigure import crop_fig
    bgr = cv2.imread(background, cv2.IMREAD_COLOR)
    bgr,_,_ = crop_fig(bgr)
    #key_detection_visual(bgr)
    frm = cv2.imread(frame, cv2.IMREAD_COLOR)
    frm,_,_ = crop_fig(frm)
    bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    frm = np.array(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
    #res = moveTowards_filter(bgr,frm)
    res_black = moveTowards_filter(bgr,frm, 30)
    res_white = moveTowards_filter(bgr,frm, 50)
    
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    print(np.shape(bgr))
    print(np.shape(res_white))

    white = np.clip(bgr.astype(np.int16)-res_white.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
    black = np.clip(res_black.astype(np.int16)-bgr.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
    cv2.imshow("result_img", white )
    cv2.waitKey(0)
    cv2.imshow("result_img", black )
    cv2.waitKey(0)
    _, thresh_white = cv2.threshold(white,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_black = cv2.threshold(black,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(src=black, thresh=30, maxval=255, type=0)
    cv2.imshow("result_img", thresh_white )
    cv2.waitKey(0)
    cv2.imshow("result_img", thresh_black )
    cv2.waitKey(0)
    cv2.destroyAllWindows()