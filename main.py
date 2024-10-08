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
from loader import Loader, Loader2
from utils import progress_bar

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
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = Loader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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

# class-balanced sampling (pseudo labeling)
def get_plabels(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    # overflow goes into remaining
    remaining = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if len(class_dict[predicted.item()]) < 100:
                class_dict[predicted.item()].append(samples[idx])
            else:
                remaining.append(samples[idx])
            progress_bar(idx, len(ploader))

    sample1k = []
    for items in class_dict.values():
        if len(items) == 100:
            sample1k.extend(items)
        else:
            # supplement samples from remaining 
            sample1k.extend(items)
            add = 100 - len(items)
            sample1k.extend(remaining[:add])
            remaining = remaining[add:]
    
    return sample1k

# confidence sampling (pseudo labeling)
## return 1k samples w/ lowest top1 score
def get_plabels2(net, samples, cycle, labeled_set_increase):
    # dictionary with 10 keys as class labels
    # class_dict = {}
    # [class_dict.setdefault(x,[]) for x in range(10)]

    # sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()])
            progress_bar(idx, len(ploader))
    top1_scores = [score.cpu().numpy() for score in top1_scores]
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:labeled_set_increase]]

# entropy sampling
def get_plabels3(net, samples, cycle):
    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[-1000:]]

def get_classdist(samples):
    # TODO: hardcode class number
    class_dist = np.zeros(2)
    for sample in samples:
        label = int(sample.split('/')[-2])
        class_dist[label] += 1
    return class_dist

if __name__ == '__main__':
    run_path = './pt4al_run'
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    
    if not os.path.exists(f"./{run_path}/_final"):
        os.makedirs(f"./{run_path}/_final")

    unlabeled_batch_size = config.unlabeled_batch_size
    unlabeled_batch_percentage_to_label = config.unlabeled_batch_percentage_to_label
    num_unlabeled_batches = config.num_unlabeled_batches
    labeled_set_increase = config.labeled_set_increase
    num_classes = config.num_classes

    labeled_images = []

    plt.figure(figsize=(10,7))
    CYCLES = num_unlabeled_batches
    for cycle in range(CYCLES):
        if not os.path.exists(f'{run_path}/cycle_{cycle}'):
            os.makedirs(f'{run_path}/cycle_{cycle}')

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device))
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Cycle ', cycle)

        # open unlabeled batch (sorted low->high)
        with open(f'./loss/batch_{cycle}.txt', 'r') as f:
            unlabeled_batch_images = f.readlines()
            
        if cycle > 0:
            print('>> Getting previous checkpoint')
            # prevnet = ResNet18().to(device)
            # prevnet = torch.nn.DataParallel(prevnet)
            checkpoint = torch.load(f'./checkpoint/main_{cycle-1}.pth')
            net.load_state_dict(checkpoint['net'])

            # sampling
            unlabeled_images_sample = get_plabels2(net, unlabeled_batch_images, cycle, labeled_set_increase)
        else:
            # first iteration: sample portion of the batch
            unlabeled_batch_images = np.array(unlabeled_batch_images)
            unlabeled_images_sample = unlabeled_batch_images[[j for j in range(0, unlabeled_batch_size, int(1/unlabeled_batch_percentage_to_label))]]
        
        # add the sampled images to the labeled set
        get_class_dist(unlabeled_images_sample, cycle, num_classes, f"./{run_path}/class_dist.txt")
        labeled_images.extend(unlabeled_images_sample)
        print(f'>> Labeled length: {len(labeled_images)}')
        trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled_images)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

        conf_matrix = None
        classification_rep = None
        for epoch in range(200):
            train(net, criterion, optimizer, epoch, trainloader)
            conf_matrix, classification_rep = test(net, criterion, epoch, cycle)
            scheduler.step()

            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.savefig(f'./{run_path}/cycle_{cycle}/confusion_matrix_{epoch}.png')
            plt.clf()

            with open(f'./{run_path}/cycle_{cycle}/metrics.txt', 'a') as f:
                f.write(str(epoch) + ' ' + classification_rep + '\n')


        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.savefig(f'./{run_path}/_final/confusion_matrix_{cycle}.png')
        plt.clf()

        with open(f'./{run_path}/_final/metrics.txt', 'a') as f:
            f.write(str(cycle) + ' ' + classification_rep + '\n')
