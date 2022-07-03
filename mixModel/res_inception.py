import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary

class conv_stack_3d(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, stride, padding):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_features, output_features, kernel_size,stride=stride,padding = padding),
            nn.BatchNorm3d(output_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x

class inception3d(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        
        self.conv_stack1 = conv_stack_3d(input_features, 64,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=0)
        self.conv_stack2 = conv_stack_3d(input_features, 96,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=0)
        self.conv_stack3 = conv_stack_3d(96, 128,
                                kernel_size=(3, 3, 3),
                                stride=(1, 1, 1),
                                padding=1)
        self.conv_stack4 = conv_stack_3d(input_features, 16,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=0)
        self.conv_stack5 = conv_stack_3d(16, 32,
                                kernel_size=(5, 5, 5),
                                stride=(1, 1, 1),
                                padding=2)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                stride=(1, 1, 1),
                                padding=1)
        self.conv_stack6 = conv_stack_3d(input_features, 32,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=0)

    def forward(self, x):
        b0 = self.conv_stack1(x)
        b1 = self.conv_stack3(self.conv_stack2(x))
        b2 = self.conv_stack5(self.conv_stack4(x))
        b3 = self.conv_stack6(self.maxpool1(x))

        x = self._shortcut3d(x, torch.cat((b0,b1,b2,b3),1))
        return x
    
    def _shortcut3d(self, input, residual):
        """3D shortcut to match input and residual and merges them with "sum"."""
        stride_dim1 = input.shape[2] // residual.shape[2]
        stride_dim2 = input.shape[3] // residual.shape[3]
        stride_dim3 = input.shape[4] // residual.shape[4]
        equal_channels = residual.shape[1] == input.shape[1]
    
        shortcut = input
        if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
            shortcut = nn.Conv3d(input.shape[1], residual.shape[1],
                kernel_size=(1, 1, 1),
                stride=(stride_dim1, stride_dim2, stride_dim3)
                )(input)
        return torch.add(shortcut, residual)
    
class res_inception3d(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.output_features = output_features
        self.conv_3772 = conv_stack_3d(input_features, 64,
                                kernel_size=(3, 7, 7),
                                stride=(2, 2, 2),
                                padding=(1, 3, 3))
        self.maxpool_3332 = nn.MaxPool3d(kernel_size=(3, 3, 3),
                                stride=(2, 2, 2),
                                padding=(1, 1, 1))
        self.conv_1111 = conv_stack_3d(64, 64,
                                kernel_size=(1, 1, 1),
                                stride=(1, 1, 1),
                                padding=(0, 0, 0))
        self.conv_3331 = conv_stack_3d(64, 192,
                                kernel_size=(3, 3, 3),
                                stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.maxpool_2222 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                stride=(2, 2, 2))
        self.averagepool_2441 = nn.AvgPool3d(kernel_size=(2, 4, 4),
                                        stride=(2, 4, 4))
        self.inception1 = inception3d(192)
        self.inception2 = inception3d(256)
        self.inception3 = inception3d(256)
        self.inception4 = inception3d(256)
        self.inception4 = inception3d(256)
        self.inception5 = inception3d(256)
        self.inception6 = inception3d(256)
        self.inception7 = inception3d(256)
        self.inception8 = inception3d(256)
        self.inception9 = inception3d(256)
        
    def forward(self, x):# [1, 512, 128, 640]
        # layer 1         
        x = self.conv_3772(x) # [64, 256, 64, 320]
        x = self.maxpool_3332(x) # [64, 128, 32, 160]
        x = self.conv_1111(x) # 
        x = self.conv_3331(x) # [192, 128, 32, 160]
        x = self.maxpool_3332(x) # [192, 64, 16, 80]
        # layer 2
        x = self.inception1(x) # [256, 64, 16, 80]
        x = self.inception2(x)
        x = self.maxpool_3332(x) # [256, 32, 8, 40]
        # layer 3
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.maxpool_2222(x) # [256, 16, 4, 20]
        # layer 4
        x = self.inception8(x)
        x = self.inception9(x)
        
        y = self.averagepool_2441(x) # [256, 8, 1, 5]
       
        x = self.averagepool_2441(x) # [256, 8, 1, 5]
        x = torch.flatten(x, 1) # [10240]
        x = nn.Linear(x.shape[1], self.output_features)(x) # [88]
        x = nn.Sigmoid()(x)
        
        return x,y
    
#############################

class conv_stack_2d(nn.Module):
    def __init__(self, input_features, output_features, kernel_size, stride, padding):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size,stride=stride,padding = padding),
            nn.BatchNorm2d(output_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x

class inception2d(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        
        self.conv_stack1 = conv_stack_2d(input_features, 64,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0)
        self.conv_stack2 = conv_stack_2d(input_features, 96,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0)
        self.conv_stack3 = conv_stack_2d(96, 128,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1)
        self.conv_stack4 = conv_stack_2d(input_features, 16,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0)
        self.conv_stack5 = conv_stack_2d(16, 32,
                                kernel_size=(5, 5),
                                stride=(1, 1),
                                padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1)
        self.conv_stack6 = conv_stack_2d(input_features, 32,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0)
        self._shortcutconv = nn.Conv2d(192, 256, kernel_size=(1, 1), stride=(1,1))
    def forward(self, x):
        b0 = self.conv_stack1(x)
        b1 = self.conv_stack3(self.conv_stack2(x))
        b2 = self.conv_stack5(self.conv_stack4(x))
        b3 = self.conv_stack6(self.maxpool1(x))

        x = self._shortcut2d(x, torch.cat((b0,b1,b2,b3),1))
        return x
    
    def _shortcut2d(self, input, residual):
        """3D shortcut to match input and residual and merges them with "sum"."""
        stride_dim1 = input.shape[2] // residual.shape[2]
        stride_dim2 = input.shape[3] // residual.shape[3]
        equal_channels = residual.shape[1] == input.shape[1]
        shortcut = input
        if stride_dim1 > 1 or stride_dim2 > 1 or not equal_channels:
            shortcut = self._shortcutconv(input)
        shortcut = shortcut
        return torch.add(shortcut, residual)
    
class res_inception2d(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.output_features = output_features
        self.conv_3772 = conv_stack_2d(input_features, 64,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3))
        self.maxpool_3332 = nn.MaxPool2d(kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=(1, 1))
        self.conv_1111 = conv_stack_2d(64, 64,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0))
        self.conv_3331 = conv_stack_2d(64, 192,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1))
        self.maxpool_2222 = nn.MaxPool2d(kernel_size=(2, 4),
                                stride=(2, 4))
        self.averagepool_2441 = nn.AvgPool2d(kernel_size=(4, 4),
                                        stride=(4, 4))
        self.inception1 = inception2d(192)
        self.inception2 = inception2d(256)
        self.inception3 = inception2d(256)
        self.inception4 = inception2d(256)
        self.inception4 = inception2d(256)
        self.inception5 = inception2d(256)
        self.inception6 = inception2d(256)
        self.inception7 = inception2d(256)
        self.inception8 = inception2d(256)
        self.inception9 = inception2d(256)
        self.linear = nn.Linear(512, self.output_features)
    def forward(self, x):# [1, 128, 640]
        # layer 1         
        x = self.conv_3772(x) # [64, 64, 320]
        x = self.maxpool_3332(x) # [64, 32, 160]
        x = self.conv_1111(x) # 
        x = self.conv_3331(x) # [192, 32, 160]
        # x = self.maxpool_3332(x) # [192, 16, 80]
        # layer 2
        x = self.inception1(x) # [256, 16, 80]
        x = self.inception2(x)
        x = self.maxpool_3332(x) # [256, 8, 40]
        # layer 3
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.maxpool_2222(x) # [256, 4, 20]
        # layer 4
        x = self.inception8(x)
        x = self.inception9(x)
               
        x = self.averagepool_2441(x) # [256, 1, 5]
        # y = torch.flatten(x, 1)
        x = torch.flatten(x, 1) # [10240]
        # x = self.linear(x) # [88]
        # x = nn.Sigmoid()(x)
        # return x,y   
        return x  



# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = res_inception2d(1,88).cuda()

# summary(model, input_size=(8, 1,128,640), device="cuda")