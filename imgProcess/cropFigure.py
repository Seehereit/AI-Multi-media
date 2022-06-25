import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from functools import cmp_to_key

def cmp_dist(x,y):
    return -((x[0]-x[2])**2 + (x[1]-x[3])**2 - (y[0]-y[2])**2 - (y[1]-y[3])**2)

def cmp_dist2(x,y):
    return -(x[1]+x[3]-y[1]-y[3])

def sort_points_clockwise(PointList):
    """提供一组凸四边形顶点，按照从左上角顶点开始顺时针方向排序
      :param points: Numpy矩阵，shape为(4, 2)，描述一组凸四边形顶点
      :return sorted_points: 经过排序的点
    """
    # 直接分别取四个点坐标中x和y的最小值作为最大外接矩形左上顶点
    points=np.array(PointList)  # 倾斜矩形
    outter_rect_l_t = np.append(np.min(points[::, 0]), np.min(points[::, 1]))
    # 求距离最大外接矩形左上点最近的点，作为给定四边形的左上顶点
    # 这一步应当是np.argmin(np.sqrt(np.sum(np.square(points - (x, y)), axis=1)))
    # 但是开不开算数平方根结果都一样，不是特别有必要，还浪费算力，就省了
    l_t_point_index = np.argmin(
        np.sum(np.square(points - outter_rect_l_t), axis=1))
    # 分别拿出来左上角点和剩余三点
    l_t_point = points[l_t_point_index]
    other_three_points = np.append(points[0:l_t_point_index:],
                                   points[l_t_point_index + 1::],
                                   axis=0)
    # 以x轴(此处以(1, 0)矢量视为x轴)为基准，根据剩余三点与x轴夹角角度排序
    BASE_VECTOR = np.asarray((1, 0))
    BASE_VECTOR_NORM = 1.0  # np.linalg.norm((1, 0))结果为1
    other_three_points = sorted(other_three_points,
                                key=lambda item: np.arccos(
                                    np.dot(BASE_VECTOR, item) /
                                    (BASE_VECTOR_NORM * np.linalg.norm(item))),
                                reverse=False)
    sorted_points = np.append(l_t_point.reshape(-1, 2),
                              np.asarray(other_three_points),
                              axis=0)
   # lt, rt, rb, lb = [sorted_points[0][0],sorted_points[0][1]], [sorted_points[1][0],sorted_points[1][1]],[sorted_points[2][0],sorted_points[2][1]], [sorted_points[3][0],sorted_points[3][1]]
    return sorted_points

def img_paste(image, polygon_list):
    '''
    img_path:图片的路径
    polygon_dict:位置的字典，形如{'yu':[[x1,y1],[x2,y2].....[x3,y3]]}
    save_name:保存的图片的名字
    Return:
    '''
    #创建一个和原图一样的全0数组
    im = np.zeros(image.shape[:2], dtype="uint8")
    roi_t = polygon_list
    roi_t = np.asarray(roi_t)
    roi_t = np.expand_dims(roi_t, axis=0)
    #把所有的点画出来
    cv2.polylines(im, roi_t, 1, 255,thickness=1)
    # im_line = im.copy()
    #把所有点连接起来，形成封闭区域
    cv2.fillPoly(im, roi_t, 255)
    # im = im - im_line
    mask = im
    #将连接起来的区域对应的数组和原图对应位置按位相与
    masked = cv2.bitwise_and(image, image, mask=mask)
    #cv2中的图片是按照bgr顺序生成的，我们需要按照rgb格式生成
    # fig = cv2.split(masked)
    masked = masked[[not np.all(masked[i] == 0) for i in range(masked.shape[0])], :]
    masked = masked[:, [not np.all(masked[:, i] == 0) for i in range(masked.shape[1])]]
    # cv2.imshow("mask",mask)
    # cv2.waitKey(0)
    # masked = cv2.merge([r, g, b])
    # return Image.fromarray(masked)
    return masked,mask


def Extend_line(p1,p2, x, y, flag):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    if flag == 1:
        if y1 == y2:
            return ([0, y1], [x, y2])
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1*y2-x2*y1)/(x1-x2)
            x3 = 0
            y3 = int(b)
            x4 = x
            y4 = int(k * x4 + b)
        return ([x3, y3], [x4, y4])
    else:
        if x1 == x2:
            return ([x1, 0], [x2, y])
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1 * y2 - x2 * y1) / (x1 - x2)
            y3 = 0
            x3 = int(-1*b/k)
            y4 = y
            x4 = int((y4-b)/k)
            return ([x3, y3], [x4, y4])

def crop_fig_bkr(image,mask):
    image = image[round(image.shape[0]/2):,:]
    masked = cv2.bitwise_and(image, image, mask=mask)
    #cv2中的图片是按照bgr顺序生成的，我们需要按照rgb格式生成
    # fig = cv2.split(masked)
    masked = masked[[not np.all(masked[i] == 0) for i in range(masked.shape[0])], :]
    masked = masked[:, [not np.all(masked[:, i] == 0) for i in range(masked.shape[1])]]
    return masked
    

def crop_fig(image):
    image = image[round(image.shape[0]/2):,:]
    img = cv2.GaussianBlur(image,(3,3),0)
    result = image.copy()
    # result = np.zeros(img.shape)
    (h, w) = result.shape[:2]
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    houghLine=cv2.HoughLinesP(edges,1,np.pi/180,118)

    if (houghLine is None):
        return [],[],[]
    houghLine = houghLine[:,0,:]

    if (len(houghLine)<2):
        return [],[],[]

    mask = np.ones(houghLine.shape)
    for x in range(len(houghLine)):
        # if ((houghLine[x][0]-houghLine[x][2])**2 + (houghLine[x][1]-houghLine[x][3])**2)<((w/100)**2+(h/100)**2):
        if (abs(houghLine[x][0]-houghLine[x][2]))<((w/60)):
            mask[x,:]=0

    houghLine = houghLine[mask.astype(bool)].reshape(-1,4)
    # import pdb;pdb.set_trace()    
    lines = sorted(houghLine, key=cmp_to_key(lambda a, b: cmp_dist(a,b)))  #这里对最后一个参数使用了经验型的值
    if (len(houghLine)<2):
        return [],[],[]

    p1,p2,p3,p4=lines[0][0:2],lines[0][2:4],lines[1][0:2],lines[1][2:4]

    (h, w) = result.shape[:2]
    if ((p1[0]==p2[0]) or (p3[0]==p4[0])):
        return [],[],[]
    # k1 = (p1[1]-p2[1])/(p1[0]-p2[0])
    # b1 = p2[1] - k1*p2[0]
    # k2 = (p3[1]-p4[1])/(p3[0]-p4[0])
    # b2 = p4[1] - k2*p4[0]
    # p1 = [0,round(b1)]
    # p2 = [w,round(w*k1+b1)]
    # p3 = [0,round(b2)]
    # p4 = [w,round(w*k2+b2)]

    p1,p2=Extend_line(p1,p2,w,h,1)
    p3,p4=Extend_line(p3,p4,w,h,1)

    result, crop_mask = img_paste(result,sort_points_clockwise([p1,p2,p4,p3]))
    result = result.copy()
    (h, w) = result.shape[:2]

    edges = cv2.Canny(result, 50, 150, apertureSize = 3)
    
    houghLine = cv2.HoughLinesP(edges,1,np.pi/180,118)
    
    if (houghLine is None):
        return [],[],[]
    houghLine = houghLine[:,0,:]

    if (len(houghLine)<2):
        return [],[],[]
    # mask = np.ones(houghLine.shape)
    # for x in range(len(houghLine)):
    #     # if ((houghLine[x][0]-houghLine[x][2])**2 + (houghLine[x][1]-houghLine[x][3])**2)<((w/60)**2+(h/60)**2):
    #     if (abs(houghLine[x][0]-houghLine[x][2]))<((w/60)):
    #         mask[x,:]=0
    # houghLine = houghLine[mask.astype(bool)].reshape(-1,4)
    
    lines = sorted(houghLine, key=cmp_to_key(lambda a, b: cmp_dist(a,b)))  #这里对最后一个参数使用了经验型的值
    lines_crop = []
    for idx in range(min(len(lines),40)):
        line = lines[idx]
        x1,y1,x2,y2 = line
        if (y1 > round(h/5)) and (y2 > round(h/5)) and (y2 < round(h/5*4)) and (y2 < round(h/5*4)):
        # if (y1 > 5) and (y2 > 5) and (y2 < h-5) and (y2 < h-5):
            lines_crop.append(line)
            # cv2.line(result,(x1,y1),(x2,y2),(125,125,125),5)
    

    lines_crop = np.array(sorted(lines_crop, key=cmp_to_key(lambda a, b: cmp_dist2(a,b))))
    # import pdb;pdb.set_trace()
    if len(lines_crop)<2 or (np.array(lines_crop[:,(1,3)])).std()<=1:
        return [],[],[]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(lines_crop[:,(1,3)])
    mask = kmeans.labels_
    group1 = lines_crop[mask==0]
    group2 = lines_crop[mask==1]
    if len(group1)==0 or len(group2)==0:
        return [],[],[]
    if (group1[:,1].mean()+group1[:,3].mean()) > (group2[:,1].mean()+group2[:,3].mean()):
        keyboard_line = group2.astype(np.int32)
        white_line = group1.astype(np.int32)
    else:
        keyboard_line = group1.astype(np.int32)
        white_line = group2.astype(np.int32)
    keyboard_line = keyboard_line[np.argmax(abs(keyboard_line[:,0]-keyboard_line[:,2]))]
    white_line = white_line[np.argmax(abs(white_line[:,0]-white_line[:,2]))]
    
    p1,p2 = white_line[0:2],white_line[2:4]
    p3,p4 = keyboard_line[0:2],keyboard_line[2:4]
    p3[1] = p3[1] + 3
    p4[1] = p4[1] + 3
    # p1,p2,p3,p4=lines_crop[0][0:2],lines_crop[0][2:4],lines_crop[-1][0:2],lines_crop[-1][2:4]
    
    if ((p1[0]==p2[0]) or (p3[0]==p4[0])):
        return [],[],[]
    # k1 = (p1[1]-p2[1])/(p1[0]-p2[0])
    # b1 = p2[1] - k1*p2[0]
    # k2 = (p3[1]-p4[1])/(p3[0]-p4[0])
    # b2 = p4[1] - k2*p4[0]
    # p1 = [0,h]
    # p2 = [w,h]
    # p3 = [0,round((b2)*1.2)]
    # p4 = [w,round((w*k2+b2)*1.2)]
    
    # p1,p2=Extend_line(p1,p2,h,w,1)
    # import pdb;pdb.set_trace()
    
    pp1, pp2 = Extend_line(p3,p4,w,h,1)
    crop_result,crop_mask2 = img_paste(result,sort_points_clockwise([[0,h],[w,h],pp1,pp2]))
    # cv2.imshow("result_img", crop_result)
    # cv2.waitKey(0)
    
    upper = 0
    while upper<crop_mask.shape[0] and crop_mask[upper].sum()==0:
        upper = upper + 1
    crop_mask[upper:upper+crop_mask2.shape[0],:]=crop_mask2
        
    # cv2.imshow("result_img", result)
    # cv2.waitKey(0)

    # import pdb;pdb.set_trace()


    # p1 = [0,round(b1)]
    # p2 = [w,round(w*k1+b1)]
    # p3 = [0,h]
    # p4 = [w,h]
    pp1, pp2 = Extend_line(p1,p2,w,h,1)
    whitekey,_ = img_paste(result,sort_points_clockwise([pp1,pp2,[0,h],[w,h]]))
    
    # cv2.imshow('Result', whitekey)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return crop_result,whitekey,crop_mask


if __name__ == '__main__':
    imgPath = "testFigures_gray/4969.bmp"
    image = cv2.imread(imgPath, 0)
    image = image[round(image.shape[0]/2):,:]
    img = cv2.GaussianBlur(image,(3,3),0)
    result = image.copy()
    # result = np.zeros(img.shape)

    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = sorted(cv2.HoughLinesP(edges,1,np.pi/180,118)[:,0,:], key=cmp_to_key(lambda a, b: cmp_dist(a,b)))  #这里对最后一个参数使用了经验型的值



    p1,p2,p3,p4=lines[0][0:2],lines[0][2:4],lines[1][0:2],lines[1][2:4]

    (h, w) = result.shape[:2]

    k1 = (p1[1]-p2[1])/(p1[0]-p2[0])
    b1 = p2[1] - k1*p2[0]
    k2 = (p3[1]-p4[1])/(p3[0]-p4[0])
    b2 = p4[1] - k2*p4[0]
    p1 = [0,round(b1)]
    p2 = [w,round(w*k1+b1)]
    p3 = [0,round(b2)]
    p4 = [w,round(w*k2+b2)]

    result = img_paste(result,sort_points_clockwise([p1,p2,p4,p3]))
    cv2.imshow('Result', result)
    cv2.waitKey(0)

    result = result.copy()
    (h, w) = result.shape[:2]

    edges = cv2.Canny(result, 50, 150, apertureSize = 3)
    lines = sorted(cv2.HoughLinesP(edges,1,np.pi/180,118)[:,0,:], key=cmp_to_key(lambda a, b: cmp_dist(a,b)))  #这里对最后一个参数使用了经验型的值
    lines_crop = []
    for idx in range(min(len(lines),40)):
        line = lines[idx]
        x1,y1,x2,y2 = line
        if (y1 > round(h/5)) and (y2 > round(h/5)) and (y2 < round(h/5*4)) and (y2 < round(h/5*4)):
            lines_crop.append(line)
            # cv2.line(result,(x1,y1),(x2,y2),(125,125,125),5)
    lines_crop = sorted(lines_crop, key=cmp_to_key(lambda a, b: cmp_dist2(a,b))) 

    p1,p2,p3,p4=lines_crop[0][0:2],lines_crop[0][2:4],lines_crop[-1][0:2],lines_crop[-1][2:4]

    k1 = (p1[1]-p2[1])/(p1[0]-p2[0])
    b1 = p2[1] - k1*p2[0]
    k2 = (p3[1]-p4[1])/(p3[0]-p4[0])
    b2 = p4[1] - k2*p4[0]
    p1 = [0,h]
    p2 = [w,h]
    p3 = [0,round((b2)*1.2)]
    p4 = [w,round((w*k2+b2)*1.2)]

    crop_result = img_paste(result,sort_points_clockwise([p1,p2,p4,p3]))
    cv2.imshow('Result', crop_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    p1 = [0,round(b1)]
    p2 = [w,round(w*k1+b1)]
    p3 = [0,h]
    p4 = [w,h]
    whitekey = img_paste(result,sort_points_clockwise([p1,p2,p4,p3]))

    # cv2.imshow('Result', whitekey)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # from key_detection import blackkeys_detection
    # print(brightness(whitekey))
    # blackkey = blackkeys_detection(crop_result)
    # pdb.set_trace()

