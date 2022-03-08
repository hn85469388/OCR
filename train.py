from dataloader import DataSet
from U2NET import U2NET
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import sys
sys.setrecursionlimit(1000)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = U2NET()
    dataset = DataSet()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)
    Lossfunction = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0 )
    net.train()

    if(device == 'cuda'):
        net.cuda()
        for epoch in range (300):
            for data, label in dataloader:
                print(data)
                data = data.cuda()
                label = label.cuda()
                o0, o1, o2, o3, o4, o5, o6 = net(data)
                loss0 = Lossfunction(o0, label)
                loss1 = Lossfunction(o1, label)
                loss2 = Lossfunction(o2, label)
                loss3 = Lossfunction(o3, label)
                loss4 = Lossfunction(o4, label)
                loss5 = Lossfunction(o5, label)
                loss6 = Lossfunction(o6, label)
                loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
    else:
        net.cpu()
        for epoch in range(300):
            for data, label in dataloader:
                data = data.cpu()
                label = label.cpu()
                o0, o1, o2, o3, o4, o5, o6 = net(data)
                loss0 = Lossfunction(o0, label)
                loss1 = Lossfunction(o1, label)
                loss2 = Lossfunction(o2, label)
                loss3 = Lossfunction(o3, label)
                loss4 = Lossfunction(o4, label)
                loss5 = Lossfunction(o5, label)
                loss6 = Lossfunction(o6, label)
                loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
    torch.save(net.state_dict(), 'test.pt')
