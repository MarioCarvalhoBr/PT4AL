import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

from torchvision.transforms.functional import resize


from config import Config
config = Config()

class RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path=config.data_dir):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(f'{path}/train/*/*')
        else:
            self.img_path = glob.glob(f'{path}/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]
        
class PuzzleLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path=config.data_dir):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(f'{path}/train/*/*')
        else:
            self.img_path = glob.glob(f'{path}/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = resize(img[:, :16, :16], (32, 32))
            img2 = resize(img[:, :16, 16:], (32, 32))
            img3 = resize(img[:, 16:, :16], (32, 32))
            img4 = resize(img[:, 16:, 16:], (32, 32))
            imgs = [img1, img2, img3, img4]
            permutations = [[0,1,2,3], [0,1,3,2], [0,2,1,3], [0,2,3,1], [0,3,1,2], [0,3,2,1], [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0], [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0], [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0]]
            random.shuffle(permutations)
            return imgs[permutations[0][0]], imgs[permutations[0][1]], imgs[permutations[0][2]], imgs[permutations[0][3]], permutations[0][0], permutations[0][1], permutations[0][2], permutations[0][3]
        
        else:
            img = self.transform(img)
            img1 = resize(img[:, :16, :16], (32, 32))
            img2 = resize(img[:, :16, 16:], (32, 32))
            img3 = resize(img[:, 16:, :16], (32, 32))
            img4 = resize(img[:, 16:, 16:], (32, 32))
            imgs = [img1, img2, img3, img4]
            permutations = [[0,1,2,3], [0,1,3,2], [0,2,1,3], [0,2,3,1], [0,3,1,2], [0,3,2,1], [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0], [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0], [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0]]
            random.shuffle(permutations)
            return imgs[permutations[0][0]], imgs[permutations[0][1]], imgs[permutations[0][2]], imgs[permutations[0][3]], permutations[0][0], permutations[0][1], permutations[0][2], permutations[0][3], self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=True, transform=None, path=config.data_dir, path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list

        if self.is_train: # train
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(f'{path}/train/*/*') # for loss extraction
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
    
class Loader_Cold(Dataset):
    def __init__(self, is_train=True, transform=None, path=config.data_dir):
        unlabeled_batch_size = config.unlabeled_batch_size
        unlabeled_batch_percentage_to_label = config.unlabeled_batch_percentage_to_label

        self.is_train = is_train
        self.transform = transform
        # TODO: hardcoded path
        with open('./loss/batch_5.txt', 'r') as f:
            self.list = f.readlines()
        self.list = [self.list[i] for i in range(0, unlabeled_batch_size, int(1/unlabeled_batch_percentage_to_label))]
        if self.is_train==True: # train
            self.img_path = self.list
        else:
            self.img_path = glob.glob(f'{path}/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train ==True:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
    
class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path=config.data_dir):
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            self.img_path = glob.glob(f'{path}/train/*/*')
        else:
            self.img_path = glob.glob(f'{path}/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
