import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt


class FacadeDataset_256(Dataset):
    def __init__(self, flag, dataDir='./image256/', data_range=(0, 8)):
        assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])

        self.dataset = []
        image_names = glob.glob(dataDir + flag + '/*.jpg')
        for i in range(data_range[0], data_range[1]):
            img = Image.open(image_names[i])

            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = img.astype('float') / 128.0

            img = np.transpose(img, [2,0,1])
            img_L = img[0,::].reshape([1, img.shape[1], img.shape[2]])
            img = img[1:, ::]

            self.dataset.append((img_L, img))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.FloatTensor(img), torch.FloatTensor(label)

class FacadeDataset_504(Dataset):
    def __init__(self, flag, dataDir='./image256/', data_range=(0, 8)):
        #assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])

        self.dataset = []
        #image_names = glob.glob(dataDir + flag + '/*.jpg')
        for i in range(data_range[0], data_range[1]):
            img_tomask = Image.open('tomask/' + str(i + 1) + '.jpg')
            #img_masked = Image.open('masked/' + str(i + 1) + '.jpg')
            img_real = Image.open('realmask/' + str(i + 1) + '.jpg')

            img_tomask = cv2.resize(cv2.cvtColor(np.asarray(img_tomask), cv2.COLOR_RGB2BGR), (128, 128))
            #img_masked = cv2.resize(cv2.cvtColor(np.asarray(img_masked), cv2.COLOR_RGB2BGR), (128, 128))
            img_real = cv2.resize(cv2.cvtColor(np.asarray(img_real), cv2.COLOR_RGB2BGR), (128, 128))

            img_tomask = img_tomask.astype('float') / 256.0
            img_tomask = np.transpose(img_tomask, [2,0,1])
            #img_masked = img_masked.astype('float') / 256.0
            #img_masked = np.transpose(img_masked, [2,0,1])
            img_real = img_real.astype('float') / 256.0
            img_real = np.transpose(img_real, [2,0,1])

            #self.dataset.append((img_masked, img_real))
            #self.dataset.append((img_masked, img_masked))
            self.dataset.append((img_tomask, img_tomask))
            self.dataset.append((img_real, img_real))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.FloatTensor(img), torch.FloatTensor(label)
