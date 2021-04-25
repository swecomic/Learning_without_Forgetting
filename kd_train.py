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
import torch.optim as optim
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

from torch.multiprocessing import Pool, Process
import timm
from tqdm import tqdm

torch.cuda.set_device(0)   # or 1,2,3


class Fitter:

    def __init__(self, s_model, t_model, T, theta, device, config):
        self.config = config
        self.epoch = 1

        self.base_dir = increment_dir(f'./{config.folder}/exp')  # runs/exp1
        # self.base_dir = f'./{config.folder}'

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.s_model = s_model
        self.t_model = t_model
        self.T = T
        self.theta = theta
        self.device = device

        #Tensorboard
        self.writer = SummaryWriter(f'{self.base_dir}/tb_graph')

        #Weight decay - Regularization
        param_optimizer = list(self.s_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # define cost/loss & optimizer
        self.criterion = nn.CrossEntropyLoss()

        # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
        self.optimizer = optim.SGD(optimizer_grouped_parameters, lr=config.lr, momentum=0.9)
        # self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config.lr)
        # self.optimizer = torch.optim.Adagrad(optimizer_grouped_parameters, lr=config.lr)

        # self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.scheduler  = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=config.lr, max_lr=0.1, step_size_up=40, mode="triangular2")
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.2)

        self.log(f'Fitter prepared. Device is {self.device}')
        self.log(f'Temperature is {self.T}')
        self.log(f'Theta is {self.theta}')

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
            train_loss, train_acc = self.kd_train_one_epoch(train_loader)

            self.log(
                f'[Train. Epoch: {self.epoch}] : , summary_loss: {train_loss.avg:.5f}, accuracy: {train_acc:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            "============= Validation ============="
            t = time.time()
            val_loss, val_acc = self.kd_validation(validation_loader)

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
                self.s_model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                # self.scheduler.step(metrics=val_acc)
                self.scheduler.step()

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
    def kd_validation(self, validation_loader):
        self.s_model.eval()
        summary_loss = AverageMeter()
        correct=0
        accuracy=0
        t = time.time()

        pbar = tqdm(enumerate(validation_loader), total=len(validation_loader))
        for step, (images, targets) in pbar:

            with torch.no_grad():
                targets = targets.to(self.device, dtype=torch.long)
                images = images.to(self.device).float()

                """output from student"""
                s_outputs = self.s_model(images)
                _, predicted = torch.max(s_outputs, 1)
                batch_size = images.shape[0]

                """Task Loss - hard target loss"""
                task_loss = self.criterion(s_outputs, targets)  # task Loss
                effective_task_loss = task_loss * (1. - self.theta)

                """output from teacher"""
                with torch.no_grad():
                    t_outputs = self.t_model(images)

                """kd_loss - soft target loss"""
                soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(s_outputs / T, dim=1),
                                                                F.softmax(t_outputs / T, dim=1)) * (T * T)
                effective_soft_loss = soft_loss * self.theta

                """total_loss"""
                total_loss = effective_soft_loss + effective_task_loss
                summary_loss.update(total_loss.detach().item(), batch_size)

                correct += (predicted == targets).sum().item()
                accuracy = correct / (batch_size * (step + 1))

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
    def kd_train_one_epoch(self, train_loader):
        self.s_model.train()
        self.t_model.eval()

        summary_loss = AverageMeter()
        correct=0
        accuracy=0
        t = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, targets) in pbar:

            targets = targets.to(self.device, dtype=torch.long)
            images = images.to(self.device).float()

            """output from student"""
            s_outputs = self.s_model(images)
            _, predicted = torch.max(s_outputs, 1)
            batch_size = images.shape[0]

            """Task Loss - hard target loss"""
            task_loss = self.criterion(s_outputs, targets)  # task Loss
            effective_task_loss = task_loss * (1. - self.theta)

            """output from teacher"""
            with torch.no_grad():
                t_outputs = self.t_model(images)

            """kd_loss - soft target loss"""
            soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(s_outputs / T, dim=1),
                                                            F.softmax(t_outputs / T, dim=1)) * (T * T)
            effective_soft_loss = soft_loss * self.theta

            """total_loss"""
            total_loss = effective_soft_loss + effective_task_loss
            # print(task_loss, "__", effective_task_loss)
            # print(soft_loss, "__"kd, effective_soft_loss)
            # print(total_loss)

            self.optimizer.zero_grad()
            total_loss.backward()

            summary_loss.update(total_loss.detach().item(), batch_size)

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
        self.s_model.eval()
        torch.save({
            'model': self.s_model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'model_state_dict': self.s_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.s_model.load_state_dict(checkpoint['model_state_dict'])
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
def run_training(s_model, t_model, T, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stat(net.to('cpu'), (3, 32, 32))
    s_model.to(device)
    summary(s_model, (3, 32, 32))

    #initializer
    s_model.apply(weights_init)

    train_loader, validation_loader = cifar_dataloader(TrainGlobalConfig.batch_size, TrainGlobalConfig.num_workers)

    fitter = Fitter(s_model=s_model, t_model=t_model, T=T, theta=theta, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, validation_loader)



"============Global Parameter Setting============"
class TrainGlobalConfig:
    num_workers = 0
    batch_size = 160
    n_epochs = 80
    lr = 0.0001

    folder = 'runs'

    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='max',
        factor=0.5,
        patience=1,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=2,
        min_lr=1e-8,
        eps=1e-08
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=int, default=1, help='Temperature T')
    parser.add_argument('--theta', type=float, default=0.1, help='Lambda Value')
    opt, unknown = parser.parse_known_args()

    s_model = Student_Model()
    t_model = Teacher_Model() #pretrained

    T = opt.temp
    theta = opt.theta

    run_training(s_model, t_model, T, theta)


