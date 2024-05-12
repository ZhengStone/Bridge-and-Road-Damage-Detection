# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os

__all__ = ['MyDataset']

class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        
        data_path_label = pd.read_csv(csv_path)
        
        self.data = data_path_label.values      # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data_path, label = self.data[index]
        data_nor_path = data_path.replace('\\', '/')
        img = Image.open(data_nor_path).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data)