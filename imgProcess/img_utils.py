import os
import cv2 as cv
def savefig_keyboard(current_pwd, path,fig,overwrite):
    dir_path = current_pwd + 'testFigures_keyboard'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path) 
    fig_path = os.path.join(dir_path,path)
    if overwrite and os.path.isfile(fig_path):
        os.remove(fig_path)
    if current_pwd.split("\\")[-2] == "video_106":
        fig_crop = fig[0:64, :]
    cv.imwrite(fig_path,fig_crop)
    print("write {}".format(path))