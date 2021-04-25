import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from pathlib import Path
import os
from typing import Union, List, Dict, Any, cast

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M0', 512, 512, 512, 'M0', 512, 512, 512, 'M0']

}


class VGG(nn.Module):
    def __init__(self, vgg_name, n_classes, last_size=1):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.n_classes = n_classes
        self.gap = nn.AdaptiveAvgPool2d((last_size, last_size))
        self.classifier = nn.Sequential(
            nn.Linear(512 * last_size * last_size, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                ]
            elif x == 'M0':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    nn.Dropout(0.25)
                ]
            else:
                x = cast(int, x)
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x

        return nn.Sequential(*layers)


"======== Student Model ========"


class Student_Model(nn.Module):
    def __init__(self):
        super(Student_Model, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.25)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.25)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool2d((27, 27))
        )

        self.stage4 = nn.Sequential(
            nn.Linear(128 * 27 * 27, 500),
            nn.ReLU()
        )

        self.stage5 = nn.Sequential(
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.stage4(x)
        x = self.stage5(x)

        return x


def Teacher_Model():
    # load a pretrained teacher model
    checkpoint = torch.load('./runs/exp10/last-checkpoint.bin')

    t_model = checkpoint['model']
    t_model.load_state_dict(checkpoint['model_state_dict'])

    # for parameter in t_model.parameters():
    #     parameter.requires_grad = False

    return t_model


def tl_model(tl_type):
    checkpoint = torch.load('./runs/exp10/last-checkpoint.bin')

    pt_model = checkpoint['model']
    pt_model.load_state_dict(checkpoint['model_state_dict'])

    num_ftrs = pt_model.classifier[-1].in_features
    pt_model.classifier[-1] = nn.Linear(num_ftrs, 100)  # replace out to 100
    pt_model.features

    "find index of Conv. layer"
    conv_layer = []
    for cnt in range(len(pt_model.features)):
        for param in pt_model.features[cnt].parameters():
            conv_layer.append(cnt)
    conv_layer = list(set(conv_layer))
    print(conv_layer)

    "find index of FC layer"
    clf_layer = []
    for cnt in range(len(pt_model.classifier)):
        for param in pt_model.classifier[cnt].parameters():
            clf_layer.append(cnt)
    clf_layer = list(set(clf_layer))
    # print(clf_layer)


    if tl_type == 0:  # Feature Extraction

        for cnt in conv_layer:
            for param in pt_model.features[cnt].parameters():
                param.requires_grad = False
                print(cnt, pt_model.features[cnt], ': freeze')

        cnt = clf_layer[-1]
        for param in pt_model.classifier[cnt].parameters():
            param.requires_grad = True
            print(cnt, pt_model.classifier[cnt], ': retrain')

    if tl_type == 1:  # Retrain FC

        for cnt in conv_layer:
            for param in pt_model.features[cnt].parameters():
                param.requires_grad = False
                print(cnt, pt_model.features[cnt], ': freeze')

        for cnt in clf_layer[:-1]:
            for param in pt_model.classifier[cnt].parameters():
                param.requires_grad = True
                print(cnt, pt_model.classifier[cnt], ': Retrain')

    elif tl_type == 2:  # Fine Tuning - early conv layer
        print("")

        for param in pt_model.parameters():
            param.requires_grad = False

        for cnt in conv_layer[:8]:
            for param in pt_model.features[cnt].parameters():
                param.requires_grad = True
                print(cnt, pt_model.features[cnt], ': retrain')

        for cnt in clf_layer[:-1]:
            for param in pt_model.classifier[cnt].parameters():
                param.requires_grad = True
                print(cnt, pt_model.classifier[cnt], ': retrain')

    elif tl_type == 3:  # Fine Tuning - later conv layer
        print("")

        for param in pt_model.parameters():
            param.requires_grad = False

        for cnt in conv_layer[-12:]:
            for param in pt_model.features[cnt].parameters():
                param.requires_grad = True
                print(cnt, pt_model.features[cnt], ': retrain')

        for cnt in clf_layer[:-1]:
            for param in pt_model.classifier[cnt].parameters():
                param.requires_grad = True
                print(cnt, pt_model.classifier[cnt], ': retrain')

    elif tl_type == 4:  # Fine Tuning - all layers
        print("")

        for param in pt_model.parameters():
            param.requires_grad = True

    return pt_model



class multiNet(nn.Module):
    def __init__(self, shared_backbone, shared_gap, old_classifier, new_classifier):
        super(multiNet, self).__init__()
        self.shared_backbone = shared_backbone
        self.shared_gap = shared_gap
        self.old_classifier = old_classifier
        self.new_classifier = new_classifier

    def forward(self, x):
        x = self.shared_backbone(x)
        x = self.shared_gap(x)
        x = x.view(x.size(0), -1)
        out_old = self.old_classifier(x)
        out_new = self.new_classifier(x)

        return out_old, out_new

class comNet(nn.Module):
    def __init__(self, shared_backbone, shared_gap, classifier):
        super(comNet, self).__init__()
        self.shared_backbone = shared_backbone
        self.shared_gap = shared_gap
        self.classifier = classifier

    def forward(self, x):
        x = self.shared_backbone(x)
        x = self.shared_gap(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(m.bias)
