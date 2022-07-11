import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import *
from midi import parse_midi
import cv2
import re


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                data_dict = self.load(*input_files)
                with open(os.path.join(data_dict["image_path"],"dataset_config.json")) as f:
                    dataset_config = json.load(f)
                for e in dataset_config:
                    cur_data = {}
                    cur_data["path"] = data_dict["path"]
                    cur_data["audio"] = data_dict["audio"][dataset_config[e]["audio_begin"]:dataset_config[e]["audio_end"]]
                    assert len(cur_data["audio"]) == SEQUENCE_LENGTH
                    cur_data["label"] = data_dict["label"][(dataset_config[e]["audio_begin"]//HOP_LENGTH):(dataset_config[e]["audio_end"]//HOP_LENGTH), :]
                    cur_data["velocity"] = data_dict["velocity"][(dataset_config[e]["audio_begin"]//HOP_LENGTH):(dataset_config[e]["audio_end"]//HOP_LENGTH), :]
                    cur_data["image_path"] = data_dict["image_path"]
                    cur_data["audio_begin"], cur_data["audio_end"] = dataset_config[e]["audio_begin"], dataset_config[e]["audio_end"]
                    if cur_data["audio"].shape[0]>=81920:
                        self.data.append(cur_data)

    def __getitem__(self, index):
        data = self.data[index]         
        result = dict(path=data['path'])

        if self.sequence_length is not None:        #327680
            # 没图像的部分传一张全黑的进去

            #黑键 白键图 上下叠在一起 每张图64*640
            # batch * 640 * 128 * 640             
            image_path = data["image_path"]
            # print("current image path is {}".format(image_path))
            result['image'] = []
            for i in range(data["audio_begin"], data["audio_end"], HOP_LENGTH):     #640张图
                with open(image_path + "\\fps.json", "r") as f:
                    fps = json.load(f)
                cur_num = i * fps["fps"] // SAMPLE_RATE 
                for e in range(0, 2):       #如果一张图找不到，至多找2次
                    mix_name = "{}\\mix\\{:>03d}.bmp".format(image_path, cur_num + e)
                    if os.path.exists(mix_name):
                        mix = cv2.imread(mix_name, cv2.IMREAD_GRAYSCALE)
                        mix = torch.tensor(cv2.resize(mix, (320, 64))).float()
                        break
                    elif e == 1:
                        # print("image %d not exist, replaced with full zero" % cur_num)
                        mix = torch.zeros((64, 320)).float()
                #mix = torch.ShortTensor(mix.reshape((1, 128, 640))).to(self.device)
                result['image'].append(mix.reshape((1, 64, 320))) 
            result['image'] = torch.cat(result['image'], dim=0).to(self.device)
            
            #512张图
            # for i in range(image_begin, image_end + 1):

            #     #rgb_name = "{}\\testFigures_keyboard\\{:>03d}.bmp".format(image_path, i)
            #     #print(rgb_name)
            #     #rgb = cv2.imread(rgb_name)
            #     #rgb = torch.ShortTensor(rgb)
            #     #print(type(rgb))
            #     #print(np.shape(rgb))
            #     #result['rgb'].append(rgb)
            #     black_name = "{}\\black\\{:>03d}.bmp".format(image_path, i)
            #     black = cv2.imread(black_name, cv2.IMREAD_GRAYSCALE)
            #     white_name = "{}\\white\\{:>03d}.bmp".format(image_path, i)
            #     white = cv2.imread(white_name, cv2.IMREAD_GRAYSCALE)
            #     #print(np.shape(black))
            #     img = np.concatenate((black, white), axis=1)
            #     #print(np.shape(img))
            #     img = torch.ShortTensor(img).to(self.device)
            #     result['image'].append(img)
                
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        result['image'] = result['image'].div_(255.0)
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path, image_path = None):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt') 
        if os.path.exists(saved_data_path):     #存在返回true
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        #tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel
        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity, image_path=image_path)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('\\flac\\', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))

class SIGHT(PianoRollAudioDataset):
    def __init__(self, path='mixModel/data/SIGHT', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, data_path=None):
        self.data_path = data_path
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)
        

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if self.data_path is not None:
            videos = self.data_path
        else:
            videos = glob(os.path.join('mixModel/data/SIGHT', 'video', 'video_*.mp4'))       
        flacs = [v.replace('\\video\\', '\\flac\\').replace('/video/', '/flac/').replace('.mp4', '.flac') for v in videos]
        tsvs = [v.replace('\\video\\', '\\tsv\\').replace('/video/', '/tsv/').replace('video', 'audio').replace('.mp4', '.tsv') for v in videos]
        image_paths = [v.replace('\\video\\', '\\image\\').replace('/video/', '/image/').replace('.mp4', '') for v in videos]
        #image 文件夹
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))
        assert(all(os.path.isfile(video) for video in videos))
        for image_path in image_paths:
            dataset_config = {}
            current_pwd = image_path
            path = sorted(os.listdir(os.path.join(current_pwd, "mix")),key = lambda i:int(re.match(r'(\d+)',i).group()))[0]
            image_begin = int(path.split('.')[0])
            image_end = sorted(os.listdir(os.path.join(current_pwd, "mix")),key = lambda i:int(re.match(r'(\d+)',i).group()))[-1]
            videoCapture = cv2.VideoCapture(current_pwd.replace("\\image\\", "\\video\\").replace('/image/', '/video/') + ".mp4")	# 读取视频文件
            fps = videoCapture.get(cv2.CAP_PROP_FPS)	# 计算视频的帧率
            with open(os.path.join(current_pwd, 'fps.json'), 'w') as f:
                data = {}
                data["fps"] = int(fps)
                json.dump(data, f)
            audio_end =  int(image_end.split('.')[0]) * SAMPLE_RATE // fps 
            audio_begin = int(image_begin * SAMPLE_RATE // fps)
            audio = soundfile.read(current_pwd.replace("\\image\\", "\\flac\\").replace('/image/', '/flac/') + ".flac", dtype='int16')[0]
            dict_num = 0
            for cur_num in range(audio_begin, len(audio), SAMPLE_INTERVAL):
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
                
        return sorted(zip(flacs, tsvs, image_paths))   