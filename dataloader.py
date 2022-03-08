from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import cv2

data_dir = r'D:/SideProject/DUTS/DUTS-TR-Image/'
label_dir =r'D:/SideProject/DUTS/DUTS-TR-Mask/'

class DataSet(Dataset):
    def __init__(self):
        super(DataSet, self).__init__()
        self.dataset = os.listdir(data_dir)
        self.dataset = self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            # imagePath = self.dataset[index]
            image = Image.open(os.path.join(data_dir, self.dataset[index]))
            label = Image.open(os.path.join(label_dir, self.dataset[index]))
            pad = max(image, label)
            size = (pad, pad)
            transform = transforms.Compose([
                transforms.CenterCrop(size),
                transforms.Resize(490),
                transforms.ToTensor()
            ])
            imagedata = transform(image)
            labeldata = transform(label)
            # image = cv2.imread(imagePath)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mask = cv2.imread()

            return imagedata,labeldata

        except Exception as e:
            print(e)

            # return self.__getitem__(index + 1)