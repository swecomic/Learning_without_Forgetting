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
from copy import deepcopy
from torch.multiprocessing import Pool, Process
import timm
from tqdm import tqdm



class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 1

        self.base_dir = increment_dir(f'./{config.folder}/exp')  # runs/exp1
        # self.base_dir = f'./{config.folder}'

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        # Old Model
        self.old_model = model
        # New Model
        self.model = deepcopy(self.old_model)

        # Replace new model's output
        in_features = self.model.classifier[2].in_features
        new_fc = nn.Linear(in_features, 100)
        self.model.classifier[2] = new_fc

        print(self.model)

        self.device = device

        #Tensorboard
        self.writer = SummaryWriter(f'{self.base_dir}/tb_graph')

        #Weight decay - Regularization
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # define cost/loss & optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.log(f'Runs # : {self.base_dir}')
        self.log(f'Fitter prepared. Device is {self.device}')
        self.log(f'Epoch is {opt.epoch}')
        self.log(f'Batch size is {opt.batch_size}')
        self.log(f'Learning rate is {opt.lr}')

    "========================================="
    "============= Model Fitting ============="
    "========================================="
    def fit(self, train_loader, validation_loader):

        start = time.time()
        # initialize recorders
        nv_stats_recorder = NVStatsRecorder(gpu_index=0)
        # start recorders
        nv_stats_recorder.start(interval=1)

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                # self.log(f'\n{timestamp}\nLR: {lr}')

            "============= Training ============="
            t = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader)

            self.log(
                f'[Train. Epoch: {self.epoch}] : , summary_loss: {train_loss.avg:.5f}, accuracy: {train_acc:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            "============= Validation ============="
            t = time.time()
            val_loss, val_acc = self.validation(validation_loader)

            self.log(
                f'[Val. Epoch: {self.epoch}] : , summary_loss: {val_loss.avg:.5f}, accuracy: {val_acc:.5f}, time: {(time.time() - t):.5f}')

            "============= Tensorboard ============="
            #Tensorboard
            self.writer.add_scalars(f'loss info', {
                'train_loss': train_loss.avg,
                'val_loss': val_loss.avg,
            }, self.epoch)

            self.writer.add_scalars(f'acc info', {
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, self.epoch)

            "============= Checkpoint ============="
            if val_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_acc)
                # self.scheduler.step()

            self.epoch += 1

        torch.cuda.current_stream().synchronize()
        self.log(
            f'Total Time: {time.time()-start}')

        # stop recorders
        nv_stats_recorder.stop()
        # get data from recorders
        gpu_data = nv_stats_recorder.get_data()
        nv_stats_recorder.plot_gpu_util(smooth=3)

    "========================================="
    "=============Validation One Epoch ============="
    "========================================="
    def validation(self, validation_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        correct=0
        accuracy=0
        t = time.time()

        pbar = tqdm(enumerate(validation_loader), total=len(validation_loader))
        for step, (images, targets) in pbar:

            with torch.no_grad():
                # targets = torch.from_numpy(to_categorical(targets, 10)).to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                images = images.to(self.device).float()
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                batch_size = images.shape[0]

                loss = self.criterion(outputs, targets)
                summary_loss.update(loss.detach().item(), batch_size)

                correct += (predicted == targets).sum().item()
                accuracy = correct / (batch_size*(step+1))

            pbar.set_postfix_str(
                f'Val Step {step}/{len(validation_loader)}, '
                f'summary_loss: {summary_loss.avg:.5f}, '
                f'accuracy: {accuracy:.5f}, '
                f'time: {(time.time() - t):.5f}'
            )

        return summary_loss, accuracy

    "========================================="
    "============= Train One Epoch ============="
    "========================================="
    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        correct=0
        accuracy=0
        t = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, targets) in pbar:

            targets = targets.to(self.device, dtype=torch.long)
            images = images.to(self.device).float()
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            correct += (predicted == targets).sum().item()
            accuracy = correct / (batch_size*(step+1))


            if self.config.step_scheduler:
                self.scheduler.step()

            pbar.set_postfix_str(
                    f'Train Step {step}/{len(train_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'accuracy: {accuracy:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}'
            )

        return summary_loss, accuracy

    def save(self, path):
        self.model.eval()
        torch.save({
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+', encoding='utf-8') as logger:
            logger.write(f'{message}\n')


"========================================="
"============ Training ============"
"========================================="
def run_training(net, flag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stat(net.to('cpu'), (3, 32, 32))
    net.to(device)
    summary(net, (3, 32, 32))

    if flag == 0:
        #initializer
        net.apply(weights_init)
        train_loader, validation_loader = cifar_dataloader(TrainGlobalConfig.batch_size, TrainGlobalConfig.num_workers)
    elif flag == 1:
        train_loader, validation_loader = face_dataloader(TrainGlobalConfig.batch_size, TrainGlobalConfig.num_workers)

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, validation_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int, default=0, help='0: Normal training 1: transfer learning')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--epoch', type=int, default=30, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--tl_type', type=int, default=1, help='transfer learning type')
    opt, unknown = parser.parse_known_args()


    if opt.flag == 0:
        net = VGG('VGG16', 10, last_size=5)
    elif opt.flag == 1:
        net = tl_model(opt.tl_type)

    "============Global Parameter Setting============"
    class TrainGlobalConfig:
        num_workers = opt.num_workers
        batch_size = opt.batch_size
        n_epochs = opt.epoch
        lr = opt.lr


        folder = 'runs'

        verbose = True
        verbose_step = 1

        step_scheduler = False  # do scheduler.step after optimizer.step
        validation_scheduler = True  # do scheduler.step after validation stage loss

        SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = dict(
            mode='max',
            factor=0.2,
            patience=2,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08
        )

    run_training(net, opt.flag)


