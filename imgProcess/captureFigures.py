import os
import cv2 as cv

def capture_figures(video_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path) 
    capture = cv.VideoCapture(video_path)
    index = 0
    while capture.isOpened():
        ret,frame = capture.read()#frame是BGR格式
        if not ret:
            print('loss this frame')
        if frame is None:
            break
        # print(frame.shape) #(795, 1055, 3)
        # cv.imshow('frame',frame)
        save_frame = "{}/{:>03d}.bmp".format(save_path, index)
        cv.imwrite(save_frame,frame)
        index = index+1
    capture.release()
    # cv.destroyAllWindows()
