import os
import math
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _setup_parser():
    pass

def load_example_pretrainig_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    classes = dataset.classes
    return dataset, classes

def train_val_split(data_dir):
    dataset, classes = load_example_pretrainig_data(data_dir)
    data_len = dataset.__len__()
    train, val = random_split(dataset, [math.ceil(0.8 * data_len), math.ceil(0.2 * data_len)])
    return train, val, classes