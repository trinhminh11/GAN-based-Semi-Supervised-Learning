import os

import torch
from torch import Tensor
from torch.utils.data import Dataset
import config
from torchvision.datasets import MNIST, CIFAR10
import numpy as np

import torchvision.transforms as tt


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DOODLE:
    TRAIN_PER_CLASS = 6000
    TEST_PER_CLASS = 1000
    def __init__(self, root: str, train: bool = True):
        self.root = root + "/DOODLE/"

        self.num_classes = len(os.listdir(self.root))

        self.train = train
        if train:
            self.num_data = self.TRAIN_PER_CLASS * self.num_classes
        else:
            self.num_data = self.TEST_PER_CLASS * self.num_classes

        self.data = np.empty([self.num_data, 28, 28], np.int64)
        self.targets = np.empty([self.num_data], dtype=np.int8)

        self.classes = []

        self.load_data()
    
    def load_data(self):
        for i, file in enumerate(os.listdir(self.root)):
            data: np.ndarray = np.load(self.root + file)
            if self.train:
                data = data[: self.TRAIN_PER_CLASS]
                n = self.TRAIN_PER_CLASS
            else:
                data = data[self.TRAIN_PER_CLASS: self.TRAIN_PER_CLASS + self.TEST_PER_CLASS]
                n = self.TEST_PER_CLASS
            
            self.data[i*n: (i+1)*n] = data
            
            target = file[18: -4]

            self.targets[i*n: (i+1)*n] = np.full([n], i)

            self.classes.append(f'{i} - {target}')


class CustomDataSet(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform = None) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        
        self.n_samples = len(x)
        self.transform = transform

    def __getitem__(self, index):
        x_sample = self.x[index]
        y_sample = self.y[index]

        return x_sample, y_sample
    
    def __len__(self):
        return self.n_samples

def supervised_samples(X: Tensor, y: Tensor, n_samples, n_classes, get_unsup = False):
    X = Tensor(X)
    y = Tensor(y)
    X_sup, y_sup = Tensor().type_as(X), Tensor().type_as(y)
    if get_unsup:
        X_unsup, y_unsup = Tensor().type_as(X), Tensor().type_as(y)

    if n_samples == -1:
        n_samples = X.shape[0]

    if n_samples == X.shape[0]:
        if get_unsup:
            return X, y, Tensor(), Tensor()
        else:
            return X, y

    n_per_class = n_samples//n_classes

    for i in range(n_classes):
        X_with_class = X[y == i]
        idx = torch.randperm(len(X_with_class))

        sup_idx = idx[:n_per_class]

        X_sup = torch.cat((X_sup, X_with_class[sup_idx]))
        y_sup = torch.cat((y_sup, Tensor([i]*len(sup_idx)).type_as(y)))

        if get_unsup:
            unsup_idx = idx[n_per_class:]
            X_unsup = torch.cat((X_unsup, X_with_class[unsup_idx]))
            y_unsup = torch.cat((y_unsup, Tensor([i]*len(unsup_idx)).type_as(y)))
    
    if get_unsup:
        return X_sup, y_sup, X_unsup, y_unsup
    else:
        return X_sup.numpy(), y_sup.numpy()


def load_data(train_transform = None, test_transform = None):
    resize32 = tt.Resize(32)

    ldict = {}
    
    test_dataset: Dataset
    exec(f'train_dataset = {config.USED_DATA}(config.DATA_DIR, train = True)', globals().update(locals()), ldict)
    exec(f'test_dataset = {config.USED_DATA}(config.DATA_DIR, train = False)', globals().update(locals()), ldict)
    train_dataset: Dataset = ldict['train_dataset']
    test_dataset: Dataset = ldict['test_dataset']
    
    X_train = np.array(resize32(torch.Tensor(train_dataset.data)), dtype=np.float64)
    X_test = np.array(resize32(torch.Tensor(test_dataset.data)), dtype=np.float64)

    
    Xmax = X_train.max()
    deno = 1

    if Xmax > 1:
        deno = 255
    
    X_train = X_train / (deno)
    X_test = X_test / (deno)

    X_train = (X_train -0.5 )/ 0.5
    X_test = (X_test - 0.5) / 0.5

    y_train = np.array(train_dataset.targets, dtype=np.int8)
    y_test = np.array(test_dataset.targets, dtype=np.int8)

    return X_train, y_train, X_test, y_test, train_dataset.classes