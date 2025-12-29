import torch
import torch.nn as nn
from torchinfo import summary

BATCH_SIZE = 1
IMAGE_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 1000
ALPHA = 1

class FirstConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, 
                              out_channels=int(32*ALPHA),    #(1)
                              kernel_size=3,    #(2)
                              stride=2,         #(3)
                              padding=1,        #(4)
                              bias=False)       #(5)
        self.bn = nn.BatchNorm2d(num_features=int(32*ALPHA))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x
    
# Codeblock 5
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):  #(1)
        super().__init__()
        
        in_channels  = int(in_channels*ALPHA)    #(2)
        out_channels = int(out_channels*ALPHA)   #(3)       
        
        if downsample:    #(4)
            stride = 2
        else:
            stride = 1
        
        self.dwconv = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels,     #(5)
                                kernel_size=3,                #(6)
                                stride=stride,                #(7)
                                padding=1,
                                groups=in_channels,           #(8)
                                bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=in_channels)   #(9)
        
        self.pwconv = nn.Conv2d(in_channels=in_channels,   
                                out_channels=out_channels,    #(10)
                                kernel_size=1,                #(11)
                                stride=1,                     #(12)
                                padding=0,                    #(13)
                                groups=1,                     #(14)
                                bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)  #(15)
        
        self.relu = nn.ReLU()    #(16)

    def forward(self, x):
        
        x = self.relu(self.bn0(self.dwconv(x)))

        x = self.relu(self.bn1(self.pwconv(x)))

        return x

# Codeblock 8a
class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.first_conv = FirstConv()    #(1)
        
        self.depthwise_sep_conv0 = DepthwiseSeparableConv(in_channels=32, 
                                                          out_channels=64)
        
        self.depthwise_sep_conv1 = DepthwiseSeparableConv(in_channels=64, 
                                                          out_channels=128, 
                                                          downsample=True)
        
        self.depthwise_sep_conv2 = DepthwiseSeparableConv(in_channels=128, 
                                                          out_channels=128)
        
        self.depthwise_sep_conv3 = DepthwiseSeparableConv(in_channels=128, 
                                                          out_channels=256, 
                                                          downsample=True)
        
        self.depthwise_sep_conv4 = DepthwiseSeparableConv(in_channels=256, 
                                                          out_channels=256)
        
        self.depthwise_sep_conv5 = DepthwiseSeparableConv(in_channels=256, 
                                                          out_channels=512, 
                                                          downsample=True)
        
        self.depthwise_sep_conv6 = nn.ModuleList(
            [DepthwiseSeparableConv(in_channels=512, out_channels=512) for _ in range(5)]
        )
        
        self.depthwise_sep_conv7 = DepthwiseSeparableConv(in_channels=512, 
                                                          out_channels=1024, 
                                                          downsample=True)
        
        self.depthwise_sep_conv8 = DepthwiseSeparableConv(in_channels=1024,  #(2)
                                                          out_channels=1024)
        
        num_out_channels = self.depthwise_sep_conv8.pwconv.out_channels      #(3)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))      #(4)
        self.fc = nn.Linear(in_features=num_out_channels,           #(5)
                            out_features=NUM_CLASSES)
        self.softmax = nn.Softmax(dim=1)                            #(6)
        
    def forward(self, x):
        x = self.first_conv(x)
        print(f"after first_conv\t\t: {x.shape}")
        
        x = self.depthwise_sep_conv0(x)
        print(f"after depthwise_sep_conv0\t: {x.shape}")
        
        x = self.depthwise_sep_conv1(x)
        print(f"after depthwise_sep_conv1\t: {x.shape}")
        
        x = self.depthwise_sep_conv2(x)
        print(f"after depthwise_sep_conv2\t: {x.shape}")
        
        x = self.depthwise_sep_conv3(x)
        print(f"after depthwise_sep_conv3\t: {x.shape}")
        
        x = self.depthwise_sep_conv4(x)
        print(f"after depthwise_sep_conv4\t: {x.shape}")
        
        x = self.depthwise_sep_conv5(x)
        print(f"after depthwise_sep_conv5\t: {x.shape}")
        
        for i, layer in enumerate(self.depthwise_sep_conv6):
            x = layer(x)
            print(f"after depthwise_sep_conv6 #{i}\t: {x.shape}")
        
        x = self.depthwise_sep_conv7(x)
        print(f"after depthwise_sep_conv7\t: {x.shape}")
        
        x = self.depthwise_sep_conv8(x)
        print(f"after depthwise_sep_conv8\t: {x.shape}")
        
        x = self.avgpool(x)
        print(f"after avgpool\t\t\t: {x.shape}")
        
        x = torch.flatten(x, start_dim=1)
        print(f"after flatten\t\t\t: {x.shape}")
        
        x = self.fc(x)
        print(f"after fc\t\t\t: {x.shape}")
        
        x = self.softmax(x)
        print(f"after softmax\t\t\t: {x.shape}")
        
        return x

# Codeblock 9
mobilenetv1 = MobileNetV1()
x = torch.randn((BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

out = mobilenetv1(x)

# Codeblock 10
mobilenetv1 = MobileNetV1()
summary(mobilenetv1, input_size=(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))