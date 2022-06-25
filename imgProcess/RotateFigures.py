from cmath import pi
import cv2
import numpy as np  
import pdb
import math






# image = cv2.imread("testFigures_gray/219.bmp", 0)
def rotate_figure(image):
    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), 90-angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
    
    image = image[round(image.shape[0]/2):,:]
    img = cv2.GaussianBlur(image,(3,3),0)
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,118)[:,0,:]
    result = image.copy()
    # result = np.zeros(img.shape)
    # pdb.set_trace()

    for idx in range(min(lines.shape[0],5)):
        line = lines[idx]
        rho = line[0] #第一个元素是距离rho
        theta = line[1] #第二个元素是角度theta
        # pdb.set_trace()
        # cv2.line(result,(x1,y1),(x2,y2),(255,255,255),3)
        print(rho)
        print(theta)
        if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
                    #该直线与第一行的交点
            pt1 = (int(rho/np.cos(theta)),0)
            #该直线与最后一行的焦点
            pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
            #绘制一条白线
            cv2.line( result, pt1, pt2, (125,125,125), 5)
        else: #水平直线
            # 该直线与第一列的交点
            pt1 = (0,int(rho/np.sin(theta)))
            #该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            #绘制一条直线
            cv2.line(result, pt1, pt2, (125,125,125), 5)


    theta = lines[0][1] * 180 / pi
    result = rotate_bound(result,theta)
    return result

# cv2.imshow('Canny', edges )
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np 
 
# img = cv2.imread("/home/sunny/workspace/images/road.jpg")
 
# img = cv2.GaussianBlur(img,(3,3),0)
# edges = cv2.Canny(img, 50, 150, apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,118)
# result = img.copy()
 
#经验参数
# minLineLength = 200
# maxLineGap = 15
# lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,0),4)
 
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

