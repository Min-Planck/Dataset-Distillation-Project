import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
import torchvision.transforms as transforms

DATASETS = {
        'cifar10': (CIFAR10, 'image'),
        'fmnist': (FashionMNIST, 'image'),
        'cifar100': (CIFAR100, 'image'),
    }

def get_transform(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name in ['fmnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return None
    
def load_data(dataset_name, batch_size=64):
    if dataset_name not in DATASETS.keys():
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    transform = get_transform(dataset_name)
    dataset_class, datatype = DATASETS[dataset_name]

    if dataset_name in ['cifar10', 'cifar100']:
        trainset = dataset_class("data", train=True, download=True, transform=transform)
        testset = dataset_class("data", train=False, download=True, transform=transform)
    else:
        trainset = dataset_class("data", train=True, download=True, transform=transform)
        testset = dataset_class("data", train=False, download=True, transform=transform)

    print(f"Loaded {dataset_name} dataset - {datatype}, with {len(trainset)} training samples and {len(testset)} test samples.")
    return trainset, testset

