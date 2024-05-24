# cold start ex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
from math import ceil

from models import *
from utils import progress_bar
from loader import Loader, Loader2

from config import Config
config = Config()

from class_dist import get_class_dist
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = Loader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device))
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/main_{cycle}.pth')
        best_acc = acc
    return confusion_matrix(all_targets, all_preds), classification_report(all_targets, all_preds, zero_division=1)


if __name__ == "__main__":
    run_path = "./random_run"
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    num_unlabeled_batches = config.num_unlabeled_batches
    labeled_set_increase = config.labeled_set_increase
    num_classes = config.num_classes

    labeled_images = []
        
    # open unlabeled batch (sorted low->high)
    with open(f'./rotation_loss.txt', 'r') as f:
        unlabeled_batch_images = f.readlines()
    
    img_paths = []

    for j in unlabeled_batch_images:
        loss_str_split = j[:-1].split('@')
        img_paths.append(loss_str_split[1]+'\n')

    random.shuffle(img_paths)
    unlabeled_batch_images = np.array(img_paths)


    CYCLES = num_unlabeled_batches
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        # sample withouth replacement from unlabeled batch_images
        unlabeled_images_sample = unlabeled_batch_images[:labeled_set_increase]
        unlabeled_batch_images = unlabeled_batch_images[labeled_set_increase:]
                
        # add the sampled images to the labeled set
        get_class_dist(unlabeled_images_sample, cycle, num_classes, f"./{run_path}/class_dist.txt")
        labeled_images.extend(unlabeled_images_sample)
        print(f'>> Labeled length: {len(labeled_images)}')

        trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled_images)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
        conf_matrix = None
        classification_rep = None
        for epoch in range(start_epoch, start_epoch+200):
            train(net, criterion, optimizer, epoch, trainloader)
            conf_matrix, classification_rep = test(net, criterion, epoch, cycle)
            scheduler.step()

        plt.figure(figsize=(10,7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.savefig(f'./{run_path}/confusion_matrix_{cycle}.png')

        with open(f'./{run_path}/metrics.txt', 'a') as f:
            f.write(str(cycle) + ' ' + classification_rep + '\n')
