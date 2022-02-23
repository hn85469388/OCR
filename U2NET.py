import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class Convalution(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,dirate = 1):
        super(Convalution,self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3 ,padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

def upsample(src, tar):
    return F.upsample(src, size=tar.shape[2:], mode='bilinear')

class Encode(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(Encode,self).__init__()
        self.conv = Convalution(in_ch, out_ch, dirate)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.conv(x))

class RSU1(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU1, self).__init__()

        self.en1 = Convalution(in_ch, out_ch, dirate=1)
        self.en2 = Encode(out_ch,mid_ch,1)
        self.en3 = Encode(mid_ch,mid_ch,1)
