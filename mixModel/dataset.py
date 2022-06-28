from asyncio.windows_events import NULL
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
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]         
        result = dict(path=data['path'])

        if self.sequence_length is not None:        #327680
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH 
            # 没图像的部分传一张全黑的进去
            n_steps = self.sequence_length // HOP_LENGTH    # 640
            step_end = step_begin + n_steps
            #黑键 白键图 上下叠在一起 每张图64*640
            # batch * 640 * 128 * 640             
            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            image_path = data['image_path']
            #image_length = audio_length / SAMPLE_RATE * FPS
            # image_begin = begin // SAMPLE_RATE * FPS
            # image_end = image_begin + self.sequence_length // SAMPLE_RATE * FPS
            #result['rgb'] = []
            mix_list = []
            for i in range(begin, end, HOP_LENGTH):     #640张图
                cur_num = i * FPS // SAMPLE_RATE 
                for e in range(0, 3):       #如果一张图找不到，至多找3次
                    mix_name = "{}\\mix\\{:>03d}.bmp".format(image_path, cur_num + e)
                    if os.path.exists(mix_name):
                        mix = cv2.imread(mix_name, cv2.IMREAD_GRAYSCALE)
                        mix = cv2.resize(mix, (320, 64))
                        break
                    elif e == 2:
                        print("image %d not exist, replaced with full zero" % cur_num)
                        mix = np.zeros((64, 320), dtype=np.uint8)
                #mix = torch.ShortTensor(mix.reshape((1, 128, 640))).to(self.device)
                mix_list.append(mix.reshape((1, 64, 320))) 
            image = torch.Tensor(np.concatenate(mix_list, axis=0)).to(self.device)
            result['image'] = image
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
                
            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

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

    def load(self, audio_path, tsv_path, image_path = NULL):
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
        flacs = [v.replace('\\video\\', '\\flac\\').replace('.mp4', '.flac') for v in videos]
        tsvs = [v.replace('\\video\\', '\\tsv\\').replace('video', 'audio').replace('.mp4', '.tsv') for v in videos]
        image_paths = [v.replace('\\video\\', '\\image\\').replace('.mp4', '') for v in videos]
        #image 文件夹
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))    
        assert(all(os.path.isfile(video) for video in videos))    
        for image_path in image_paths:
            if os.path.exists(image_path):      #如果文件夹存在则视为已经图像处理完成，所以别动文件夹里的文件
                continue
            # else:
                # os.makedirs(image_path)
                # current_pwd = image_path + "\\"
                # video = image_path.replace('\\image\\', '\\video\\') + '.mp4'
                # capture_figures(video, current_pwd + 'testFigures')
                # #后面的部分放到load函数里

                # bgr = get_background(current_pwd)
                # testfigures = glob(os.path.join(current_pwd, 'testFigures', '*.bmp'))
                # for item in testfigures:
                #     os.remove(item)
                # os.removedirs(current_pwd + 'testFigures')
                # bgr = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                # for path in sorted(os.listdir(current_pwd + "testFigures_keyboard"),key = lambda i:int(re.match(r'(\d+)',i).group())):
                #     frame = cv2.imread(os.path.join(current_pwd + "testFigures_keyboard",path) , cv2.IMREAD_COLOR)
                #     # import pdb;pdb.set_trace()
                #     frm = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                #     if len(bgr[0])!=len(frm[0]):
                #         continue
                #     res_black = moveTowards_filter(bgr,frm, 30)
                #     res_white = moveTowards_filter(bgr,frm, 50)
                #     white = np.clip(bgr.astype(np.int16)-res_white.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
                #     white = cv2.resize(white, (640, 64))
                #     os.makedirs(current_pwd + 'white', exist_ok=True)
                #     cv2.imwrite(current_pwd + 'white\\' + path, white)
                #     black = np.clip(res_black.astype(np.int16)-bgr.astype(np.int16),a_min=0,a_max=1).astype(np.uint8)*255
                #     white = cv2.resize(white, (640, 64))
                #     os.makedirs(current_pwd + 'black', exist_ok=True)
                #     cv2.imwrite(current_pwd + 'black\\' + path, black)
                #     mix = np.concatenate((black, white), axis=0)
                #     os.makedirs(current_pwd + 'mix', exist_ok=True)
                #     cv2.imwrite(current_pwd + 'mix\\' + path, mix)                    
                
        return sorted(zip(flacs, tsvs, image_paths))   