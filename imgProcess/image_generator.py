import os, cv2, re
from glob import glob
import numpy as np
import soundfile
from captureFigures import capture_figures
from main import get_background
from backgroundFilter import moveTowards_filter
import json
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from mixModel.constants import FPS, SAMPLE_RATE, SEQUENCE_LENGTH

videos = glob(os.path.join('mixModel/data/SIGHT', 'video', 'video_*.mp4'))
image_paths = [v.replace('\\video\\', '\\image\\').replace('.mp4', '') for v in videos]
#image 文件夹

for image_path in image_paths:
    print("当前工作目录为：{}".format(image_path))
    if not os.path.exists(image_path):      
        os.makedirs(image_path)
    current_pwd = image_path + "\\"
    video = image_path.replace('\\image\\', '\\video\\') + '.mp4'
    
    if not os.path.exists(current_pwd + 'testFigures_keyboard'):     #生成键盘图片集
        print("检测到testFigures_keyboard文件夹缺失")
        print("生成testFigures文件")
        capture_figures(video, current_pwd + 'testFigures')
        #后面的部分放到load函数里
        print("生成testFigures_keyboard文件")
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

    if not os.path.exists(current_pwd + 'black'):       #生成黑键白键差值图像
        print("检测到black和white文件夹缺失")
        input("暂停等待手动删除testFigures_keyboard中首尾没有键盘的图片，按Enter继续：")
        print("开始生成black和white文件")
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

    if not os.path.exists(current_pwd + 'mix'):     #将黑白差值拼在一起
        print("检测到mix文件夹缺失，生成mix文件")
        for path in sorted(os.listdir(current_pwd + "white"),key = lambda i:int(re.match(r'(\d+)',i).group())):
            white = cv2.imread(os.path.join(current_pwd + "white",path), cv2.COLOR_BGR2GRAY)
            white = cv2.resize(white, (640, 64))
            black = cv2.imread(os.path.join(current_pwd + "black",path), cv2.COLOR_BGR2GRAY)
            black = cv2.resize(black, (640, 64))
            mix = np.concatenate((black, white), axis=0)
            os.makedirs(current_pwd + 'mix', exist_ok=True)
            cv2.imwrite(current_pwd + 'mix\\' + path, mix) 
    
#写一个json文件，dataset_config, 记录每一段,取5秒160帧，读取的信息存在json，字典的内容还是一个字典，记录音频起始终止，图片起始终止
#第一个数据点是第一张图片，每一个数据点是在前面基础加sequence_length // 2 ，audio全零并且结束位置在audio总长度的后十分之一
    if not os.path.exists(current_pwd + 'dataset_config.json'):
        print("检测到dataset_config.json文件缺失")
        dataset_config = {}
        path = sorted(os.listdir(current_pwd + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group()))[0]
        image_begin = int(path.split('.')[0])
        image_end = sorted(os.listdir(current_pwd + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group()))[-1]
        audio_end = int(image_end.split('.')[0]) * SAMPLE_RATE // FPS
        audio_begin = image_begin * SAMPLE_RATE // FPS
        audio = soundfile.read(current_pwd.replace("\\image\\", "\\flac\\")[0:-1] + ".flac", dtype='int16')[0]
        dict_num = 0
        for cur_num in range(audio_begin, len(audio), SEQUENCE_LENGTH // 2):
            #if audio[cur_num] == 0 and cur_num >= len(audio) * 9 // 10:     # 用最后一张图片作为终止条件会不会好一点？
            if cur_num + SEQUENCE_LENGTH >= audio_end:
                break
            sampling = {}
            sampling["audio_begin"] = cur_num
            sampling["audio_end"] = cur_num + SEQUENCE_LENGTH
            #sampling["image_begin"] = cur_num * FPS // SAMPLE_RATE
            #sampling["image_end"] = sampling['audio_end'] * FPS // SAMPLE_RATE
            dataset_config[dict_num] = sampling
            dict_num += 1
        with open(current_pwd + 'dataset_config.json', 'w') as f:
            json.dump(dataset_config, f)

        



        

# 取数据改成均匀采样