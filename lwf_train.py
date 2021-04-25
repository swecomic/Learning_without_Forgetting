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

from torch.multiprocessing import Pool, Process
import timm
from tqdm import tqdm
from copy import deepcopy
torch.cuda.set_device(0)   # or 1,2,3


class Fitter:

    def __init__(self, model,  T, theta, device, config):
        self.config = config
        self.epoch = 1

        self.base_dir = increment_dir(f'./{config.folder}/exp')  # runs/exp1
        # self.base_dir = f'./{config.folder}'

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        #old model
        checkpoint = torch.load('./runs/exp10/last-checkpoint.bin')
        self.old_model = checkpoint['model']
        self.old_model.load_state_dict(checkpoint['model_state_dict'])

        #new_model
        self.model = model

        self.T = T
        self.theta = theta
        self.device = device

        print(self.model)

        #Tensorboard
        self.writer = SummaryWriter(f'{self.base_dir}/tb_graph')

        # define cost/loss & optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.log(f'Runs # : {self.base_dir}')
        self.log(f'Fitter prepared. Device is {self.device}')
        self.log(f'Epoch is {opt.epoch}')
        self.log(f'Batch size is {opt.batch_size}')
        self.log(f'Learning rate is {opt.lr}')
        self.log(f'Learning rate is {opt.wlr}')

    "========================================="
    "============= Model Fitting ============="
    "========================================="
    def fit(self, train_loader, validation_loader):

        start = time.time()
        "========================================="
        "============= Warm-up - Start  ============="
        "========================================="

        # Weight decay - Regularization
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.wlr)
        scheduler = self.config.SchedulerClass(optimizer, **self.config.scheduler_params)

        for e in range(self.config.w_n_epochs):

            "============= Training ============="
            t = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, scheduler)

            self.log(
                f'[Train. Epoch: {self.epoch}] : , summary_loss: {train_loss.avg:.5f}, accuracy: {train_acc:.5f}, time: {(time.time() - t):.5f}')
            # self.save(f'{self.base_dir}/last-checkpoint.bin')

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
            # if val_loss.avg < self.best_summary_loss:
            #     self.best_summary_loss = val_loss.avg
            #     self.model.eval()
            #     self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
            #     for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
            #         os.remove(path)

            if self.config.validation_scheduler:
                scheduler.step(metrics=val_acc)
                # self.scheduler.step()

            self.epoch += 1

        torch.cuda.current_stream().synchronize()
        self.log(
            f'Total Time: {time.time()-start}')

        "========================================="
        "============= Warm-up - End  ============="
        "========================================="

        save(self.model, optimizer, scheduler, self.best_summary_loss, self.epoch, f'{self.base_dir}/warm.bin')

        w_checkpoint = torch.load(f'{self.base_dir}/warm.bin') #load a warmed-up mdel
        w_model = w_checkpoint['model']

        shared_backbone = w_model.features  # 𝜃s
        shared_gap = w_model.gap
        new_classifier = w_model.classifier # 𝜃n
        old_classifier = self.old_model.classifier # 𝜃o

        # Modiy model to have two classifiers
        self.multi_model = multiNet(shared_backbone, shared_gap, old_classifier, new_classifier)

        for param in self.multi_model.parameters():
            param.requires_grad = True

        print(self.multi_model)

        "========================================="
        "============= LWF Training - Start  ============="
        "========================================="
        self.best_summary_loss = 10 ** 5
        self.epoch = 1

        # Weight decay - Regularization
        param_optimizer = list(self.multi_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0005},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        scheduler = self.config.SchedulerClass(optimizer, **self.config.scheduler_params)

        for e in range(self.config.n_epochs):

            "============= Training ============="
            t = time.time()
            train_loss, train_acc = self.kd_train_one_epoch(train_loader, optimizer, scheduler)

            self.log(
                f'[Train. Epoch: {self.epoch}] : , summary_loss: {train_loss.avg:.5f}, accuracy: {train_acc:.5f}, time: {(time.time() - t):.5f}')
            save(self.multi_model, optimizer, scheduler, self.best_summary_loss, self.epoch, f'{self.base_dir}/last-checkpoint.bin')

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
                self.multi_model.eval()
                save(self.multi_model, optimizer, scheduler, self.best_summary_loss, self.epoch, f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                # scheduler.step(metrics=val_acc)
                scheduler.step()

            self.epoch += 1
        "========================================="
        "============= LWF Training - End  ============="
        "========================================="


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
    def train_one_epoch(self, train_loader, optimizer, scheduler):
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

            optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            optimizer.step()

            correct += (predicted == targets).sum().item()
            accuracy = correct / (batch_size*(step+1))


            if self.config.step_scheduler:
                scheduler.step()

            pbar.set_postfix_str(
                    f'Train Step {step}/{len(train_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'accuracy: {accuracy:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}'
            )

        return summary_loss, accuracy

    "========================================="
    "============= LWF - Validation One Epoch ============="
    "========================================="
    def kd_validation(self, validation_loader):
        self.multi_model.eval()
        self.old_model.eval()
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
                output_ot, output_nt = self.multi_model(images)
                _, predicted = torch.max(output_nt, 1)
                batch_size = images.shape[0]

                """Task Loss - hard target loss"""
                task_loss = self.criterion(output_nt, targets)  # task Loss
                effective_task_loss = task_loss * (1. - self.theta)

                """output from teacher"""
                with torch.no_grad():
                    output_oldNet = self.old_model(images)

                """kd_loss - soft target loss"""
                soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_ot / self.T, dim=1),
                                                                F.softmax(output_oldNet / self.T, dim=1)) * (
                                        self.T * self.T)
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
    "============= LWF - Train One Epoch ============="
    "========================================="
    def kd_train_one_epoch(self, train_loader, optimizer, scheduler):
        self.multi_model.train()
        self.old_model.eval()

        summary_loss = AverageMeter()
        correct=0
        accuracy=0
        t = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, targets) in pbar:

            targets = targets.to(self.device, dtype=torch.long)
            images = images.to(self.device).float()

            """output from student"""
            output_ot, output_nt = self.multi_model(images)
            _, predicted = torch.max(output_nt, 1)
            batch_size = images.shape[0]

            """Task Loss - hard target loss"""
            task_loss = self.criterion(output_nt, targets)  # task Loss
            effective_task_loss = task_loss * (1. - self.theta)

            """output from teacher"""
            with torch.no_grad():
                output_oldNet = self.old_model(images)

            """kd_loss - soft target loss"""
            soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_ot / self.T, dim=1),
                                                            F.softmax(output_oldNet / self.T, dim=1)) * (self.T * self.T)
            effective_soft_loss = soft_loss * self.theta

            """total_loss"""
            total_loss = effective_soft_loss + effective_task_loss
            # print(task_loss, "__", effective_task_loss)
            # print(soft_loss, "__"kd, effective_soft_loss)
            # print(total_loss)

            optimizer.zero_grad()
            total_loss.backward()

            summary_loss.update(total_loss.detach().item(), batch_size)

            optimizer.step()

            correct += (predicted == targets).sum().item()
            accuracy = correct / (batch_size*(step+1))


            if self.config.step_scheduler:
                scheduler.step()

            pbar.set_postfix_str(
                    f'Train Step {step}/{len(train_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'accuracy: {accuracy:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}'
            )

        return summary_loss, accuracy



    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+', encoding='utf-8') as logger:
            logger.write(f'{message}\n')

    # def load(path):
    #     checkpoint = torch.load(path)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.best_summary_loss = checkpoint['best_summary_loss']
    #     self.epoch = checkpoint['epoch'] + 1


def save(model, optimizer, scheduler, best_summary_loss, epoch, path):
    model.eval()
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_summary_loss': best_summary_loss,
        'epoch': epoch,
    }, path)




"========================================="
"============ Training ============"
"========================================="
def run_training(net, T, theta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stat(net.to('cpu'), (3, 32, 32))
    net.to(device)
    summary(net, (3, 32, 32))


    train_loader, validation_loader = face_dataloader(TrainGlobalConfig.batch_size, TrainGlobalConfig.num_workers)

    fitter = Fitter(model=net, T=T, theta=theta, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, validation_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=int, default=20, help='Temperature T')
    parser.add_argument('--theta', type=float, default=0.3, help='Lambda Value')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--epoch', type=int, default=300, help='')
    parser.add_argument('--wepoch', type=int, default=80, help='epoch for warm-up')
    parser.add_argument('--lr', type=float, default=0.00003, help='Learning Rate')
    parser.add_argument('--wlr', type=float, default=0.0003, help='Learning Rate for warm-up')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    opt, unknown = parser.parse_known_args()


    net = tl_model(1)

    "============Global Parameter Setting============"
    class TrainGlobalConfig:
        num_workers = opt.num_workers
        batch_size = opt.batch_size
        n_epochs = opt.epoch
        w_n_epochs = opt.wepoch
        lr = opt.lr
        wlr = opt.wlr


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

    run_training(net, opt.temp, opt.theta)


