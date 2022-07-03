import cv2, os, re
import numpy as np
import soundfile, json
data = {1:"sss"}
test1 = {}
test1[1] = data
print(test1)
data = {2:"bbb"}
print(test1)
path = sorted(os.listdir("mixModel/data/SIGHT\\image\\video_131\\" + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group()))[0]
print(path)
print("mixModel/data/SIGHT\\flac\\video_131\\"[0:-1] + ".flac")
audio, sr = soundfile.read("mixModel/data/SIGHT\\flac\\video_131\\"[0:-1] + ".flac", dtype='int16')
print(np.shape(audio))
print(np.sum(audio == 0))

with open("mixModel/data/SIGHT\\image\\video_131" + "\\dataset_config.json") as f:
    dataset_config = json.load(f)
for e in dataset_config:
    print(e)
#cv2.imwrite('.', fig2)