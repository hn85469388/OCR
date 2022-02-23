import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class Convalution(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,dirate = 1):
        super(Convalution,self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)