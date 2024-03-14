import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import numpy as np


# Data processor
class DataProcessing:
    def __init__(self, dataset_name, root_path='~/data/', batch_size=128, seed=88, val_ratio=0.2, augmentation = False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seed = seed
        self.val_ratio = val_ratio
        self.root_path = root_path
        self.path = self.root_path + dataset_name
        self.augmentation = augmentation

        if self.dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset_name == 'MNIST':
            self.transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2))
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((1 / 2, 1 / 2, 1 / 2), (1 / 2, 1 / 2, 1 / 2))
            ])
        else:
            raise NotImplementedError

        if self.augmentation==False:
            self.transform_train = self.transform_test

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        print('data number training set', len(indices))
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def subsample(self, dataset, indices, num_data=None):
        if num_data != None:
            indices = indices[:num_data]
        return data.Subset(dataset, indices)

    def get_dataloader(self, shuffle=True):
        if self.dataset_name == 'CIFAR10':
            dataset_train = datasets.CIFAR10(root=self.path, train=True, download=True, transform=self.transform_train)
            testset = datasets.CIFAR10(root=self.path, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == 'CIFAR100':
            dataset_train = datasets.CIFAR100(root=self.path, train=True, download=True, transform=self.transform_train)
            testset = datasets.CIFAR100(root=self.path, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == 'SVHN':
            dataset_train = datasets.SVHN(root=self.path, split='train', download=True, transform=self.transform_train)
            testset = datasets.SVHN(root=self.path,  split='test', download=True, transform=self.transform_test)
        elif self.dataset_name == 'MNIST':
            dataset_train = datasets.MNIST(root=self.path, train=True, download=True, transform=self.transform_train)
            testset = datasets.MNIST(root=self.path, train=False, download=True, transform=self.transform_test)
        else:
            raise NotImplementedError

        train_index = self.get_split_indices("train", dataset_train, int(self.val_ratio * len(dataset_train)))
        val_index = self.get_split_indices("validation", dataset_train, int(self.val_ratio * len(dataset_train)))
        trainset = self.subsample(dataset_train, train_index)
        valset = self.subsample(dataset_train, val_index)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        return trainset, valset, testset, trainloader, valloader, testloader

