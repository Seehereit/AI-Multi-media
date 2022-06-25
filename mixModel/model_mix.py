import torch
import torch.nn.functional as F
from torch import nn

from mel import melspectrogram
from res_inception import res_inception2d
from onset_and_frames import onset_and_frames

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.res_inception2d = res_inception2d(1,88)
        self.onset_and_frames = onset_and_frames(741,88)
    
    def forward(self, x, mel): # (640, 229)
        pic_data = []
        x = torch.unsqueeze(x,2)
        for i in range(x.shape[1]):
            pic_data.append(self.res_inception2d(x[:,i,:,:])[1].unsqueeze(0))
        pic_data = torch.cat(pic_data,dim=0).permute(1,0,2)
        data_concat = torch.cat((mel,pic_data),2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.onset_and_frames(data_concat)
        return onset_pred, offset_pred, frame_pred, velocity_pred
    
    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']
        images = batch['image']
        
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, frame_pred, velocity_pred = self(images,mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses
    

from torchinfo import summary
model = Net().to(device)
summary(model, input_size=[(8,160,128,640),(8, 160, 229)],device=device)