import cv2
import numpy as np
from numpy import uint8
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def blackkeys_detection(src):   #输入一张图片，输出一个二维数组，储存黑键的具体信息
    #每行是一个黑键，从第一列开始依次是左上角x坐标，左上角y坐标，外接矩形的宽，外接矩形的高，
    #连通域面积，形心x坐标，形心y坐标
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)     #转灰度
    #按OTSU阈值转为二值图像
    ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    thresh = 255 - thresh   #反转，以白键为背景
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S) #连通组件算法
    num_labels = output[0]  #连通域数量
    labels = output[1]      #具体的标签信息，和原图像同大小
    #stats储存每个连通域的具体信息，分别是左上角坐标x, y, 外接矩形宽和高，以及连通域面积
    stats = output[2] 
    centroids = output[3]   #连通域形心坐标
    data = np.empty((num_labels, 2))
    data[:, 0] = np.array(stats[:, 4])      #横坐标面积
    data[:, 1] = np.array(centroids[:, 1])  #纵坐标形心的y坐标
    maxloc = np.argmax(data[:, 0])          #面积最大的索引，一般是背景图像即白键
    data[maxloc, 0] = 0             
    tmpmax = np.max(data[:, 0])             #面积第二大
    cond = data[:, 1] >= np.shape(labels)[0] / 2    #筛选形心纵坐标大于图像高度一半，即不是黑键
    data[cond, :] = [tmpmax, np.shape(labels)[0]]   #为这些点重新赋值，以便在下一步k-means分类中与黑键区分开

    # 聚类数量
    k = 2
    # 训练模型
    model = KMeans(n_clusters=k)
    model.fit(data)
    # 分类中心点坐标
    centers = model.cluster_centers_
    # 预测结果
    result = model.predict(data)
    #按照平均面积区分出哪一类是黑键
    sum1, sum2, num1, num2 = 0, 0, 0, 0
    for i in range(0, num_labels):
        if result[i] == 0:
            sum1 += data[i, 0]
            num1 +=1
        elif result[i] == 1:
            sum2 += data[i, 0]
            num2 +=1
    avg1 = sum1 / num1
    avg2 = sum2 / num2
    if avg1 > avg2:
        classify = result.astype(bool)
    else:
        result = 1 - result
        classify = result.astype(bool)
    blackkey = stats[classify, :]
    blackkey = np.insert(blackkey, 5, centroids[classify, 0], axis = 1)
    blackkey = np.insert(blackkey, 6, centroids[classify, 1], axis = 1)
    #每行是一个黑键，从第一列开始依次是左上角x坐标，左上角y坐标，外接矩形的宽，外接矩形的高，
    #连通域面积，形心x坐标，形心y坐标
    return blackkey

def blackkeys_detection_visual(src):    #可视化，上一个函数加了几个画图
    result_img = np.zeros(np.shape(src), dtype=uint8)
    print("the shape of origin image is ", np.shape(result_img))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = 255 - thresh
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    #print(num_labels)
    colour = []
    for e in range(0, num_labels):
        colour.append([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
    print(colour[0])
    # The second cell is the label matrix
    labels = output[1]
    print("the shape of labels is " , np.shape(labels))
    #print(np.unique(labels))
    #for i in range(0, 6):
    #    print((labels==i).sum())
    for i in range(0, np.shape(labels)[0]):
        for k in range(0, np.shape(labels)[1]):
            if labels[i, k] != 0:
                result_img[i, k] =  colour[ labels[i, k] ]
    cv2.imshow("result_img", result_img)
    cv2.waitKey(0)
    # The third cell is the stat matrix
    stats = output[2]
    print("the shape of stats is ", np.shape(stats))
    # The fourth cell is the centroid matrix
    centroids = output[3]
    print("the shape of centroids is ", np.shape(centroids))
    data = np.empty((num_labels, 2))
    data[:, 0] = np.array(stats[:, 4])
    data[:, 1] = np.array(centroids[:, 1])
    plt.plot(data[:, 0], data[:, 1], 'o',color='b')
    plt.show()
    maxloc = np.argmax(data[:, 0])
    data[maxloc, 0] = 0
    tmpmax = np.max(data[:, 0])
    cond = data[:, 1] >= np.shape(labels)[0] / 2 
    data[cond, :] = [tmpmax, np.shape(labels)[0]]
    #label = np.array(range(0, num_labels))
    #print(label)
    #labelset = set(label[cond])
    plt.plot(data[:, 0], data[:, 1], 'o',color='b')
    plt.show()
    # 聚类数量
    k = 2
    # 训练模型
    model = KMeans(n_clusters=k)
    model.fit(data)
    # 分类中心点坐标
    centers = model.cluster_centers_
    # 预测结果
    result = model.predict(data)
    sum1, sum2, num1, num2 = 0, 0, 0, 0
    for i in range(0, num_labels):
        if result[i] == 0:
            sum1 += data[i, 0]
            num1 +=1
        elif result[i] == 1:
            sum2 += data[i, 0]
            num2 +=1
    avg1 = sum1 / num1
    avg2 = sum2 / num2
    print("avg1 is %d" % (sum1 / num1))
    print("avg2 is %d" % (sum2 / num2))
    if avg1 > avg2:
        classify = result.astype(bool)
    else:
        result = 1 - result
        classify = result.astype(bool)
    blackkey = stats[classify, :]
    blackkey = np.insert(blackkey, 5, centroids[classify, 0], axis = 1)
    blackkey = np.insert(blackkey, 6, centroids[classify, 1], axis = 1)
    # 用不同的颜色绘制数据点
    mark = ['or', 'og', 'ob', 'ok']
    for i, d in enumerate(data):
        plt.plot(d[0], d[1], mark[result[i]])
    # 画出各个分类的中心点
    mark = ['*r', '*g', '*b', '*k']
    for i, center in enumerate(centers):
        plt.plot(center[0], center[1], mark[i], markersize=20)
    plt.show()
    mark=np.empty((4,3))
    mark[0,:]=[255,255,255]
    mark[1,:]=[0,255,255]
    mark[2,:]=[255,0,255]
    mark[3,:]=[255,255,0]
    output = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        tmp=result[i]
        mask = labels == i
        output[:, :, 0][mask] = mark[tmp,0]
        output[:, :, 1][mask] = mark[tmp,1]
        output[:, :, 2][mask] = mark[tmp,2]
    cv2.imshow('oginal', output)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return blackkey

def whitekeys_detection(black, y_max):                   #输入黑键信息和图片y轴高度，输出白键信息
    white = np.zeros((52, 8), dtype=np.intc)             #返回值，每一行代表一个白键，前四个值代表下矩形的左上、右下坐标，后四个值代表上矩形的左上、右下坐标
    white_wid1 =  ( black[19, 5] - black[14, 5] ) / 7    #假定第20个黑键和第15个黑键的形心x坐标和白键之间的黑缝重合，以此标定白键宽度
    whitekey_encoder1 = np.array([0,0,1,1,2,3,3,4,5,6,6,7,8,8,9,10,11,11,12,13,13,14,15,16,16,17,18,18,19,20,21,21,22,23,23,24,25,26,26,27,28,28,29,30,31,31,32,33,33,34,35,-1]).reshape((52,1))
    whitekey_encoder2 = np.array([1,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,1,3,2,1,3,3,2,0]).reshape((52,1))
    #whitekey_encoder[i] = [j, k]，代表第i个白键的信息，j表示被第i个黑键所占，k=0表示不被黑键占，=1表示右边有黑键，=2表示左边有黑键，=3表示两边都有黑键
    #whitekey_encoder = np.append(whitekey_encoder1, whitekey_encoder2, axis=1)
    x = black[14, 5]                                     #以第15个黑键为起点
    y = np.min(black, axis=0)[3]                         #y代表下矩形的上边的y坐标，取黑键外接矩形最小高度
    for i in range(20, -1, -1):                          #第14个黑键左边有21个白键
        white[i, 0:4] = [int(x - white_wid1 + 1), y, int(x), y_max - 1]
        x -= white_wid1
    x = black[14, 5]
    for i in range(21, 52):                              #第14个黑键右边有31个白键
        white[i, 0:4] = [int(x + 1), y, int(x + white_wid1), y_max - 1]
        x += white_wid1
    for i in range(0, 52):
        if whitekey_encoder2[i] == 0:                    #没被黑键占的话就用下矩形的坐标
            white[i, 4:8] = [white[i, 0], 0, white[i, 2], y]
        elif whitekey_encoder2[i] == 1:                  #右边有黑键，上矩形右下角x坐标取右边黑键外接矩形的x坐标
            tmpx2 = black[whitekey_encoder1[i], 0]       
            white[i, 4:8] = [white[i, 0], 0, tmpx2, y]
        elif whitekey_encoder2[i] == 2:                  #左边有黑键，上矩形左上角x坐标取左边黑键外接矩形x坐标加上外接矩形宽度
            tmpx1 = black[whitekey_encoder1[i], 0] + black[whitekey_encoder1[i], 2]
            white[i, 4:8] = [tmpx1, 0, white[i, 2], y]
        elif whitekey_encoder2[i] == 3:                  #左右都有黑键，综合上两条
            tmpx1 = black[whitekey_encoder1[i], 0] + black[whitekey_encoder1[i], 2]
            tmpx2 = black[whitekey_encoder1[i] + 1, 0]
            white[i, 4:8] = [tmpx1, 0, tmpx2, y]
    return white

def key_detection(src):                                  #输入一张图片，输出一个和图像一样大小的二维数组，每个像素点的值代表是哪个键
    #从1到52是从左往右数52个白键，从53到88是从左往右数36个黑键
    black = blackkeys_detection(src)
    white = whitekeys_detection(black, np.shape(src)[0])
    result = np.zeros((np.shape(src)[0], np.shape(src)[1]), dtype=np.intc)
    e = 1
    for i in white:
        #result[i[1]:i[3], i[0]:i[2]] = e
        result[i[5]:i[7], i[4]:i[6]] = e
        #tmp_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]  可视化
        #src[i[1]:i[3], i[0]:i[2]] = tmp_color
        #src[i[5]:i[7], i[4]:i[6]] = tmp_color
        e+= 1
    for i in black:
        result[(i[1] + 1):(i[1] + i[3] - 1), (i[0] + 1):(i[0] + i[2] - 1)] = e
        #tmp_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]  可视化
        #src[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])] = tmp_color
        e+= 1
    return result

def key_detection_visual(src):                           #上一个函数的可视化版
    #从1到52是从左往右数52个白键，从53到88是从左往右数36个黑键
    black = blackkeys_detection_visual(src)
    white = whitekeys_detection(black, np.shape(src)[0])
    result_img1 = src
    for i in white:
        cv2.rectangle(result_img1, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 1)
        cv2.rectangle(result_img1, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 1)
    cv2.imshow("result_img1", result_img1)
    cv2.waitKey(0)
    result = np.zeros((np.shape(src)[0], np.shape(src)[1]), dtype=np.intc)
    e = 1
    result_img2 = np.zeros(np.shape(src), dtype=uint8)
    for i in white:
        result[i[1]:i[3], i[0]:i[2]] = e
        result[i[5]:i[7], i[4]:i[6]] = e
        tmp_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]  #可视化
        result_img2[i[1]:i[3], i[0]:i[2]] = tmp_color
        result_img2[i[5]:i[7], i[4]:i[6]] = tmp_color
        e+= 1
    for i in black:
        result[(i[1] + 1):(i[1] + i[3] - 1), (i[0] + 1):(i[0] + i[2] - 1)] = e
        tmp_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]  #可视化
        result_img2[(i[1] + 1):(i[1] + i[3] - 1), (i[0] + 1):(i[0] + i[2] - 1)] = tmp_color
        e+= 1
    cv2.imshow("result_img2", result_img2)
    cv2.waitKey(0)
    return result

def get_keys_visual(bgr, key_list):     
    black = blackkeys_detection(bgr)
    white = whitekeys_detection(black, np.shape(bgr)[0])
    result_img = bgr
    for e in key_list:
        if e <= 52:
            cv2.rectangle(result_img, (white[e, 0], white[e, 1]), (white[e, 2], white[e, 3]), (0, 0, 255), 1)
            cv2.rectangle(result_img, (white[e, 4], white[e, 5]), (white[e, 6], white[e, 7]), (0, 0, 255), 1)
        else:
            cv2.rectangle(result_img, (black[e - 53, 0], black[e - 53, 1]), (black[e - 53, 0] + black[e - 53, 2], black[e - 53, 1] + black[e - 53, 3]), (0, 0, 255), 1)
    cv2.imshow("result_img", result_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    src = cv2.imread('testFigures_keyboard/219.bmp', cv2.IMREAD_COLOR)
    result = key_detection_visual(src)
