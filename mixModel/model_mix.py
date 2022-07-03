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
    
    def run_res_inc(self, x):
        pic_data = []
        x = torch.unsqueeze(x,2)
        for i in range(x.shape[1]):
            pic_data.append(self.res_inception2d(x[:,i,:,:])[1].unsqueeze(0))
        return torch.cat(pic_data,dim=0).permute(1,0,2)
    
    def forward(self, x, mel): # (640, 229)
        
        x = torch.cat((mel,self.run_res_inc(x)),2)
        try:
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self.onset_and_frames(x)
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
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
    
    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

# from torchinfo import summary
# model = Net().to(device)
# summary(model, input_size=[(8,160,128,640),(8, 160, 229)],device=device)