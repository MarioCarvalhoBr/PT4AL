'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import *
from loader import Loader, RotationLoader
from utils import progress_bar

from config import Config
config = Config()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = RotationLoader(is_train=False,  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/rotation.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = loss.item()
            s = str(float(loss)) + '@' + str(path[0]) + "\n"

            with open('./rotation_loss.txt', 'a') as f:
                f.write(s)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    with open('./rotation_loss.txt', 'w') as f:
        f.write('')
    test(1)
    with open('./rotation_loss.txt', 'r') as f:
        losses = f.readlines()

    loss_1 = []
    name_2 = []

    for j in losses:
        loss_str_split = j[:-1].split('@')
        loss_1.append(loss_str_split[0])
        name_2.append(loss_str_split[1])

    batch_loss_path = np.array(loss_1)
    sort_index = np.argsort(batch_loss_path)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x) # convert to high loss first

    if not os.path.isdir('loss'):
        os.mkdir('loss')

    train_set_size = config.train_set_size
    unlabeled_batch_size = config.unlabeled_batch_size
    num_classes = config.num_classes
    num_unlabeled_batches = config.num_unlabeled_batches

    for current_batch in range(num_unlabeled_batches):
        # sample minibatch from unlabeled pool 
        batch_images = sort_index[current_batch * unlabeled_batch_size:min((current_batch + 1) * unlabeled_batch_size, train_set_size)]

        class_dist = np.zeros(num_classes)
        batch_loss_path = './loss/batch_' + str(current_batch) + '.txt'

        for img_path in batch_images:
            class_index = int(name_2[img_path].split('/')[-2])
            class_dist[class_index] +=1
            with open(batch_loss_path, 'a') as f:
                f.write(name_2[img_path]+'\n')

        with open('./loss/class_dist.txt', 'a') as f:
            f.write(str(current_batch) + ' Class Distribution: ' + str(class_dist) + '\n')
