import os

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

DATA_DIR = './data'

NUM_WORKERS = 2


class MNISTDataModule:
    def __init__(self, batch_size, val_split=True):
        super().__init__()
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32, 32))
        ])

        mnist_train = MNIST(DATA_DIR, train=True, transform=transform, download=True)
        if val_split:
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        else:
            self.mnist_train = mnist_train
        self.mnist_test = MNIST(DATA_DIR, train=False, transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=NUM_WORKERS, pin_memory=True,
                          drop_last=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=NUM_WORKERS, pin_memory=True,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=NUM_WORKERS, pin_memory=True,
                          drop_last=False)
