from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

data_dir = ''
label_dir = ''

class DataSet(Dataset):
    def __init__(self):
        super(DataSet, self).__init__()
        self.dataset = os.listdir(data_dir)
        self.dataset = self.dataset

    def __getitem(self, index):
        try:
            image = Image.open(os.path.join(data_dir, self.dataset[index])).convert('RGB')
            label = Image.open(os.path.join(label_dir, self.dataset[index])).convert('L')
            pad = max(image,label)
            size = (pad,pad)
            transform = transforms.Compose([
                transforms.CenterCrop(size),
                transforms.Resize(490),
                transforms.ToTensor()
            ])
            imagedata = transform(image)
            labeldata = transform(label)

            return imagedata,labeldata
        except:
            return self.__getitem(index+1)