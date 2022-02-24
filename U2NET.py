import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

def upsample(src, tar):
    return F.upsample(src, size=tar.shape[2:], mode='bilinear')

class Convalution(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,dirate = 1):
        super(Convalution,self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3 ,padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))



class Encode(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(Encode,self).__init__()
        self.conv = Convalution(in_ch, out_ch, dirate)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.conv(x))

class RSU1(nn.Module):
    # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU1, self).__init__()

        self.en1 = Convalution(in_ch, out_ch, dirate=1)
        self.en2 = Encode(out_ch, mid_ch, 1)
        self.en3 = Encode(mid_ch, mid_ch, 1)
        self.en4 = Encode(mid_ch, mid_ch, 1)
        self.en5 = Encode(mid_ch, mid_ch, 1)
        self.en6 = Encode(mid_ch, mid_ch, 1)
        self.en7 = Convalution(mid_ch, mid_ch, 2)
        self.en8 = Convalution(mid_ch, mid_ch, 2)

        self.de6 = Convalution(mid_ch * 2, mid_ch, dirate=1)
        self.de5 = Convalution(mid_ch * 2, mid_ch, dirate=1)
        self.de4 = Convalution(mid_ch * 2, mid_ch, dirate=1)
        self.de3 = Convalution(mid_ch * 2, mid_ch, dirate=1)
        self.de2 = Convalution(mid_ch * 2, mid_ch, dirate=1)
        self.de1 = Convalution(mid_ch * 2, mid_ch, dirate=1)

    def forward(self, x):
        en1 = self.en1(x)
        en2 = self.en2(en1)
        en3 = self.en2(en2)
        en4 = self.en2(en3)
        en5 = self.en2(en4)
        en6 = self.en2(en5)
        en7 = self.en2(en6)
        en8 = self.en2(en7)

        de6 = self.de6(torch.cat([en8, en7], dim=1))
        de5 = self.de5(torch.cat([en6, en6], dim=1))
        de5 = upsample(de5, en5)
        de4 = self.de4(torch.cat([en5, en5], dim=1))
        de4 = upsample(de4, en4)
        de3 = self.de3(torch.cat([en4, en4], dim=1))
        de3 = upsample(de3, en3)
        de2 = self.de2(torch.cat([en3, en3], dim=1))
        de2 = upsample(de2, en2)
        de1 = self.de1(torch.cat([de2,en2],  dim=1))
        de1 = upsample(de1, en1)

        return de1 + en1

class RSU2(nn.Module):
        # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
        def __init__(self, in_ch=3, mid_ch=12, out_ch=3):

            super(RSU2, self).__init__()

            self.en1 = Convalution(in_ch, out_ch, dirate=1)
            self.en2 = Encode(out_ch, mid_ch, 1)
            self.en3 = Encode(mid_ch, mid_ch, 1)
            self.en4 = Encode(mid_ch, mid_ch, 1)
            self.en5 = Encode(mid_ch, mid_ch, 1)
            self.en6 = Encode(mid_ch, mid_ch, 1)
            self.en7 = Convalution(mid_ch, mid_ch, 2)

            self.de5 = Convalution(mid_ch * 2, mid_ch)
            self.de4 = Convalution(mid_ch * 2, mid_ch)
            self.de3 = Convalution(mid_ch * 2, mid_ch)
            self.de2 = Convalution(mid_ch * 2, mid_ch)
            self.de1 = Convalution(mid_ch * 2, mid_ch)

        def forward(self, x):
            en1 = self.en1(x)
            en2 = self.en2(en1)
            en3 = self.en2(en2)
            en4 = self.en2(en3)
            en5 = self.en2(en4)
            en6 = self.en2(en5)
            en7 = self.en2(en6)



            de5 = self.de5(torch.cat([en7, en6], dim=1))
            de4 = self.de4(torch.cat([en5, en5], dim=1))
            de4 = upsample(de4, en4)
            de3 = self.de3(torch.cat([en4, en4], dim=1))
            de3 = upsample(de3, en3)
            de2 = self.de2(torch.cat([en3, en3], dim=1))
            de2 = upsample(de2, en2)
            de1 = self.de1(torch.cat([de2, en2], dim=1))
            de1 = upsample(de1, en1)

            return de1 + en1


class RSU3(nn.Module):
    # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU3, self).__init__()

        self.en1 = Convalution(in_ch, out_ch)
        self.en2 = Encode(out_ch, mid_ch, 1)
        self.en3 = Encode(mid_ch, mid_ch, 1)
        self.en4 = Encode(mid_ch, mid_ch, 1)
        self.en5 = Convalution(mid_ch, mid_ch)
        self.en6 = Convalution(mid_ch, mid_ch, 2)



        self.de4 = Convalution(mid_ch * 2, mid_ch)
        self.de3 = Convalution(mid_ch * 2, mid_ch)
        self.de2 = Convalution(mid_ch * 2, mid_ch)
        self.de1 = Convalution(mid_ch * 2, mid_ch)

    def forward(self, x):
        en1 = self.en1(x)
        en2 = self.en2(en1)
        en3 = self.en2(en2)
        en4 = self.en2(en3)
        en5 = self.en2(en4)
        en6 = self.en2(en5)

        de4 = self.de4(torch.cat([en6, en5], dim=1))
        de3 = self.de3(torch.cat([en4, en4], dim=1))
        de3 = upsample(de3, en3)
        de2 = self.de2(torch.cat([en3, en3], dim=1))
        de2 = upsample(de2, en2)
        de1 = self.de1(torch.cat([de2, en2], dim=1))
        de1 = upsample(de1, en1)

        return de1 + en1

class RSU4(nn.Module):
        # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
        def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
            super(RSU4, self).__init__()

            self.en1 = Convalution(in_ch, out_ch)
            self.en2 = Encode(out_ch, mid_ch, 1)
            self.en3 = Encode(mid_ch, mid_ch, 1)
            self.en4 = Convalution(mid_ch, mid_ch)
            self.en5 = Convalution(mid_ch, mid_ch, 2)



            self.de3 = Convalution(mid_ch * 2, mid_ch)
            self.de2 = Convalution(mid_ch * 2, mid_ch)
            self.de1 = Convalution(mid_ch * 2, mid_ch)

        def forward(self, x):
            en1 = self.en1(x)
            en2 = self.en2(en1)
            en3 = self.en2(en2)
            en4 = self.en2(en3)
            en5 = self.en2(en4)


            de3 = self.de3(torch.cat([en5, en4], dim=1))
            de2 = self.de2(torch.cat([en3, en3], dim=1))
            de2 = upsample(de2, en2)
            de1 = self.de1(torch.cat([de2, en2], dim=1))
            de1 = upsample(de1, en1)

            return de1 + en1

class RSU5(nn.Module):
        # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
        def __init__(self, in_ch=3, mid_ch=12, out_ch=3):

            super(RSU5, self).__init__()

            self.en1 = Convalution(in_ch, out_ch)
            self.en2 = Convalution(out_ch, mid_ch)
            self.en3 = Convalution(mid_ch, mid_ch, dirate=2)
            self.en4 = Convalution(mid_ch, mid_ch, dirate=4)
            self.en5 = Convalution(mid_ch, mid_ch, dirate=8)


            self.de3 = Convalution(mid_ch * 2, mid_ch, dirate=4)
            self.de2 = Convalution(mid_ch * 2, mid_ch, dirate=2)
            self.de1 = Convalution(mid_ch * 2, out_ch)

        def forward(self, x):
            en1 = self.en1(x)
            en2 = self.en2(en1)
            en3 = self.en2(en2)
            en4 = self.en2(en3)

            de3 = self.de3(torch.cat([en4, en3], dim=1))
            de2 = self.de2(torch.cat([en3, en3], dim=1))
            de1 = self.de1(torch.cat([de2, en2], dim=1))

            return de1 + en1