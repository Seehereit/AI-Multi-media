import cv2
import numpy as np
fig1 = np.ones((64, 320))
fig1 = fig1 * 255
cv2.imshow("fig1", fig1)
cv2.waitKey(0)
path = "mixModel/data/SIGHT/image/video_100/mix"
test1 = cv2.imread(path + "/4900.bmp", cv2.IMREAD_GRAYSCALE)
test2 = cv2.resize(test1, (320, 64))
print(np.shape(test2))  
cv2.imshow("test2", test2)
cv2.waitKey(0)
test1 = cv2.resize(test1, (64, 320))
print(np.shape(test1))
cv2.imshow("test1", test1)
cv2.waitKey(0)
#cv2.imwrite('.', fig1)
fig2 = np.ones((320, 64))
fig2 = fig2 * 255
cv2.imshow("fig2", fig2)
cv2.waitKey(0)
#cv2.imwrite('.', fig2)