import argparse
import time
import sys
import os
from datetime import datetime
import glob
import numpy as np
from copy import copy
from nvstatsrecorder.recorders import NVStatsRecorder

import torch
import torchvision
import torch.utils.data as utils
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchstat import stat
from torchsummary import summary
#
#
# import sys
# sys.path.append('/data/jiylee/tmp/pycharm_project_145/vision_assignment_202010')
from helper import *
from model import *
from dataset import *

from torch.multiprocessing import Pool, Process
import timm
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def old_validation(model, validation_loader, model_type):
    model.eval()
    summary_loss = AverageMeter()
    correct = 0
    accuracy = 0
    t = time.time()

    pbar = tqdm(enumerate(validation_loader), total=len(validation_loader))
    for step, (images, targets) in pbar:
        with torch.no_grad():
            # targets = torch.from_numpy(to_categorical(targets, 10)).to(self.device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            images = images.to(device).float()

            if model_type == 'lwf':
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            batch_size = images.shape[0]

            loss = criterion(outputs, targets)
            summary_loss.update(loss.detach().item(), batch_size)

            correct += (predicted == targets).sum().item()
            accuracy = correct / (batch_size * (step + 1))

        pbar.set_postfix_str(
            f'Val Step {step}/{len(validation_loader)}, '
            f'summary_loss: {summary_loss.avg:.5f}, '
            f'accuracy: {accuracy:.5f}, '
            f'time: {(time.time() - t):.5f}'
        )

    return summary_loss, accuracy



train_loader, validation_loader = cifar_dataloader(32, 0)


"====== Old Task Performance - Feature Extraction ======"
#
checkpoint = torch.load('./runs/exp34/best-checkpoint-141epoch.bin')
fe_model = checkpoint['model']
fe_model.load_state_dict(checkpoint['model_state_dict'])
num_ftrs = fe_model.classifier[-1].in_features
fe_model.classifier[-1] = nn.Linear(num_ftrs, 10)  # replace out to 10

val_loss, val_acc = old_validation(fe_model.to(device), validation_loader, 'fe')


"====== Performance - Fine Tune  ======"

checkpoint = torch.load('./runs/exp35/best-checkpoint-070epoch.bin')
ft_model = checkpoint['model']
ft_model.load_state_dict(checkpoint['model_state_dict'])
num_ftrs = ft_model.classifier[-1].in_features
ft_model.classifier[-1] = nn.Linear(num_ftrs, 10)  # replace out to 10

val_loss, val_acc = old_validation(ft_model.to(device), validation_loader, 'ft')


"====== Performance - LwF old task ======"

checkpoint = torch.load('./runs/exp46/last-checkpoint.bin')
lwf_model = checkpoint['model']
lwf_model.load_state_dict(checkpoint['model_state_dict'])
val_loss, val_acc = old_validation(lwf_model.to(device), train_loader, 'lwf')







