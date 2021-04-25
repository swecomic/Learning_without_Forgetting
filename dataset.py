
import torchvision
import torch.utils.data as utils
import torchvision.transforms as transforms
from model import *


def cifar_dataloader(batch_size, num_workers):

    # data augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    "CIFAR10 Data"
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validation_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=utils.RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=utils.SequentialSampler(validation_dataset),
        pin_memory=False,
        drop_last = True
    )

    return train_loader, validation_loader


def face_dataloader(batch_size, num_workers):

    # data augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    "face Data"
    train_dataset = torchvision.datasets.ImageFolder(os.path.join("./face/downsized", 'train'), transform=transform_train)
    validation_dataset = torchvision.datasets.ImageFolder(os.path.join("./face/downsized", 'test'), transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=utils.RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=utils.SequentialSampler(validation_dataset),
        pin_memory=False,
        drop_last = True
    )

    return train_loader, validation_loader