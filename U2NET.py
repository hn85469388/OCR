import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

def upsample(src, tar):
    return F.upsample(src, size=tar.shape[2:], mode='bilinear')

class Convalution(nn.Module):
    def __init__(self, in_ch=3, out_ch=3,dirate = 1):
        super(Convalution,self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class Encode(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(Encode, self).__init__()
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
        self.en7 = Convalution(mid_ch, mid_ch, 1)
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
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        en5 = self.en5(en4)
        en6 = self.en6(en5)
        en7 = self.en7(en6)
        en8 = self.en8(en7)

        de6 = self.de6(torch.cat([en8, en7], dim=1))
        de5 = self.de5(torch.cat([de6, en6], dim=1))
        de5 = upsample(de5, en5)
        de4 = self.de4(torch.cat([de5, en5], dim=1))
        de4 = upsample(de4, en4)
        de3 = self.de3(torch.cat([de4, en4], dim=1))
        de3 = upsample(de3, en3)
        de2 = self.de2(torch.cat([de3, en3], dim=1))
        de2 = upsample(de2, en2)
        de1 = self.de1(torch.cat([de2, en2],  dim=1))
        de1 = upsample(de1, en1)

        return de1 + en1

class RSU2(nn.Module):
        # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
        def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
            super(RSU2, self).__init__()

            self.en1 = Convalution(in_ch, out_ch)
            self.en2 = Encode(out_ch, mid_ch, 1)
            self.en3 = Encode(mid_ch, mid_ch, 1)
            self.en4 = Encode(mid_ch, mid_ch, 1)
            self.en5 = Encode(mid_ch, mid_ch, 1)
            self.en6 = Convalution(mid_ch, mid_ch, dirate=1)
            self.en7 = Convalution(mid_ch, mid_ch, dirate=2)

            self.de5 = Convalution(mid_ch * 2, mid_ch, dirate=1)
            self.de4 = Convalution(mid_ch * 2, mid_ch, dirate=1)
            self.de3 = Convalution(mid_ch * 2, mid_ch, dirate=1)
            self.de2 = Convalution(mid_ch * 2, mid_ch, dirate=1)
            self.de1 = Convalution(mid_ch * 2, mid_ch, dirate=1)

        def forward(self, x):
            en1 = self.en1(x)
            en2 = self.en2(en1)
            en3 = self.en3(en2)
            en4 = self.en4(en3)
            en5 = self.en5(en4)
            en6 = self.en6(en5)
            en7 = self.en7(en6)



            de5 = self.de5(torch.cat([en7, en6], dim=1))
            de4 = self.de4(torch.cat([de5, en5], dim=1))
            de4 = upsample(de4, en4)
            de3 = self.de3(torch.cat([de4, en4], dim=1))
            de3 = upsample(de3, en3)
            de2 = self.de2(torch.cat([de3, en3], dim=1))
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
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        en5 = self.en5(en4)
        en6 = self.en6(en5)

        de4 = self.de4(torch.cat([en6, en5], dim=1))
        de3 = self.de3(torch.cat([de4, en4], dim=1))
        de3 = upsample(de3, en3)
        de2 = self.de2(torch.cat([de3, en3], dim=1))
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
            en3 = self.en3(en2)
            en4 = self.en4(en3)
            en5 = self.en5(en4)


            de3 = self.de3(torch.cat([en5, en4], dim=1))
            de2 = self.de2(torch.cat([de3, en3], dim=1))
            de2 = upsample(de2, en2)
            de1 = self.de1(torch.cat([de2, en2], dim=1))
            de1 = upsample(de1, en1)

            return de1 + en1

class RSU5(nn.Module):
        # in_ch and out_ch are input and output chaneel number, mid_ch is hidden layer chanel num
        def __init__(self, in_ch=3, mid_ch=12, out_ch=3):

            super(RSU5, self).__init__()

            self.en1 = Convalution(in_ch, out_ch)
            self.en2 = Convalution(out_ch, mid_ch, dirate=1)
            self.en3 = Convalution(mid_ch, mid_ch, dirate=2)
            self.en4 = Convalution(mid_ch, mid_ch, dirate=4)
            self.en5 = Convalution(mid_ch, mid_ch, dirate=8)


            self.de3 = Convalution(mid_ch * 2, mid_ch, dirate=4)
            self.de2 = Convalution(mid_ch * 2, mid_ch, dirate=2)
            self.de1 = Convalution(mid_ch * 2, out_ch, dirate=1)

        def forward(self, x):
            en1 = self.en1(x)
            en2 = self.en2(en1)
            en3 = self.en3(en2)
            en4 = self.en4(en3)

            de3 = self.de3(torch.cat([en4, en3], dim=1))
            de2 = self.de2(torch.cat([de3, en2], dim=1))
            de1 = self.de1(torch.cat([de2, en1], dim=1))

            return de1 + en1
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.En_1 = nn.Sequential(RSU1(in_ch, 32, 64),
                                  nn.MaxPool2d(2, 2, ceil_mode=True))

        self.En_2 = nn.Sequential(RSU2(64, 32, 128),
                                  nn.MaxPool2d(2, 2, ceil_mode=True))

        self.En_3 = nn.Sequential(RSU3(128, 64, 256),
                                  nn.MaxPool2d(2, 2, ceil_mode=True))

        self.En_4 = nn.Sequential(RSU4(256, 128, 512),
                                  nn.MaxPool2d(2, 2, ceil_mode=True))

        self.En_5 = nn.Sequential(RSU5(512, 256, 512),
                                  nn.MaxPool2d(2, 2, ceil_mode=True))

        self.En_6 = RSU5(512, 256, 512)

        self.De_5 = RSU5(1024, 256, 512)
        self.De_4 = RSU4(1024, 128, 256)
        self.De_3 = RSU3(512, 64, 128)
        self.De_2 = RSU2(256, 32, 64)
        self.De_1 = RSU1(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # self.sup0 = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        en1 = self.En_1(x)
        en2 = self.En_2(en1)
        en3 = self.En_3(en2)
        en4 = self.En_4(en3)
        en5 = self.En_5(en4)
        en6 = self.En_6(en5)

        de5 = self.De_5(torch.cat([en6, en5],dim=1))
        de5 = upsample(de5, en4)
        de4 = self.De_4(torch.cat([de5, en4], dim=1))
        de4 = upsample(de4, en3)
        de3 = self.De_3(torch.cat([de4, en3], dim=1))
        de3 = upsample(de3, en2)
        de2 = self.De_2(torch.cat([de3, en2], dim=1))
        de2 = upsample(de2, en1)
        de1 = self.De_1(torch.cat([de2, en1], dim=1))
        de1 = upsample(de1, x)

        side1 = self.side1(de1)
        side2 = self.side2(de2)
        side3 = self.side3(de3)
        side4 = self.side4(de4)
        side5 = self.side5(de5)
        side6 = self.side6(en6)

        sup1 = side1
        sup2 = upsample(side2, x)
        sup3 = upsample(side3, x)
        sup4 = upsample(side4, x)
        sup5 = upsample(side5, x)
        sup6 = upsample(side6, x)
        sup0 = self.sup0(torch.cat([sup1, sup2, sup3, sup4, sup5, sup6], dim=1))
        return F.sigmoid(sup0), F.sigmoid(sup1), F.sigmoid(sup2), F.sigmoid(sup3), F.sigmoid(sup4), F.sigmoid(sup5), \
               F.sigmoid(sup6)

if __name__ == '__main__':
    x = torch.rand((1, 3, 512,512))
    m=U2NET()




