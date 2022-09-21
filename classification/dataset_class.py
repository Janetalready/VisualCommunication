import torch.utils.data as data
import os
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import random
import pickle

class Dataset(data.Dataset):

    def __init__(self, split_root, data_root, train=True):
        self.split_root = split_root
        self.data_root = data_root

        if train:
            with open(os.path.join(self.data_root, 'train_imgs.p'), 'rb') as f:
                self.imgs = pickle.load(f)

            with open(os.path.join(self.data_root, 'train_y.p'), 'rb') as f:
                self.cates = pickle.load(f)
            print(len(self.imgs))
        else:
            with open(os.path.join(self.data_root, 'test_imgs.p'), 'rb') as f:
                self.imgs = pickle.load(f)

            with open(os.path.join(self.data_root, 'test_y.p'), 'rb') as f:
                self.cates = pickle.load(f)
            print(len(self.imgs))
        self.width = 128
        self.width_r = 128

    def __getitem__(self, index):

        img = self.imgs[index]
        img_s = torch.tensor(img).float()*255
        img_s = img_s.unsqueeze(0)
        # img_s = torch.tensor(img).float()

        y = self.cates[index]
        return img_s, int(y)

    def __len__(self):
        return len(self.imgs)
