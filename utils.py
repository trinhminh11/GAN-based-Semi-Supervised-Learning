import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import config
from torchvision.datasets import MNIST, CIFAR10

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def mapping(value, s1, e1, s2, e2):
	return s2 + (e2 - s2) * ((value-s1)/(e1-s1))

def one_hot(y):
	one_hot_y = torch.zeros((len(y), 10))
	one_hot_y[torch.arange(len(y)), y] = 1
	return one_hot_y

def supervised_samples(X: Tensor, y: Tensor, n_samples, n_classes):
	X_sup, y_sup = Tensor().type_as(X), Tensor().type_as(y)

	if n_samples == -1:
		n_samples = X.shape[0]

	n_per_class = n_samples//n_classes
	

	for i in range(n_classes):
		X_with_class = X[y == i]
		ix = torch.randint(0, len(X_with_class), [n_per_class])

		X_sup = torch.cat((X_sup, X_with_class[ix]))
		y_sup = torch.cat((y_sup, Tensor([i]*n_per_class).type_as(y)))


	return X_sup, y_sup

def to_device(data: Tensor, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)

def load_data(start = 0, end = 1):
	if config.USED_DATA == "MNIST":
		data = MNIST(config.DATA_DIR, train = True, download=True)
		X_train = data.data.unsqueeze(1).float()

		test_data = MNIST(config.DATA_DIR, train = False, download=True)
		X_test = test_data.data.unsqueeze(1).float()

	if config.USED_DATA == "CIFAR10":
		data = CIFAR10(config.DATA_DIR, train = True, download=True)
		X_train = Tensor(data.data)
		X_train = X_train.permute(0, 3, 1, 2)

		test_data = CIFAR10(config.DATA_DIR, train = False, download=True)
		X_test = Tensor(test_data.data)
		X_test = X_test.permute(0, 3, 1, 2)


	X_train = mapping(X_train, 0, 255, start, end)
	y_train = Tensor(data.targets)

	X_test = mapping(X_test, 0, 255, start, end)
	y_test = Tensor(test_data.targets)

	
	return X_train, y_train, X_test, y_test, data.classes

def print_config():
	
	variables = [(name, value) for name, value in vars(config).items() if (name[:2] != "__")]

	max_name_len = len(max(variables, key=lambda x: len(x[0]))[0])
	max_val_len = len(str(max(variables, key=lambda x: len(str(x[1])))[1]))

	for name, value in variables:
		print(f'{name:<{max_name_len}}:  {str(value):>{max_val_len}}')

def ceil(x):
	if x == int(x):
		return int(x)
	else:
		return int(x)+1

def calc_mean_std(images: torch.Tensor):
	B, C, W, H = images.shape
	n = B*W*H

	mean = [0]*C
	std = [0]*C

	for image in images:
		for i in range(C):
			mean[i] += image[i].sum().item()

	for i in range(C):
		mean[i] /= n

	for image in images:
		for i in range(C):
			std[i] += ((image[i] - mean[i]).pow(2)).sum()


	for i in range(C):
		std[i] = torch.sqrt(std[i]/n).item()

	return mean, std

class CustomDataSet(Dataset):
	def __init__(self, x: Tensor, y: Tensor, transform = None) -> None:
		self.x: Tensor = x
		self.y: Tensor = y
		
		self.n_samples = len(y)

		self.transform = transform


	def __getitem__(self, index):
		x_sample = self.x[index]
		y_sample = self.y[index]
		if self.transform:
			x_sample = self.transform(x_sample).type_as(x_sample)

		return x_sample, y_sample
	
	def __len__(self):
		return self.n_samples
	
class DeviceDataLoader():
	def __init__(self, dl: DataLoader, device):
		self.dl = dl
		self.device = device
		
	def __iter__(self):
		for b in self.dl: 
			yield to_device(b, self.device)

	def __len__(self):
		return len(self.dl)

	def num_data(self):
		return len(self.dl.dataset)

def plotting(history, sched_lr = False):
	train_loss = history['train_loss']
	val_loss = history['val_loss']
	train_acc = history['train_acc']
	val_acc = history['val_acc']
	
	epochs = len(train_loss)


	if sched_lr:
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 5))

		ax3.plot(np.arange(len(history['lrs'])), history['lrs'])

		ax3.set_xlabel('batches')
		ax3.set_ylabel('learning rate')

		batch_length = len(history['lrs']) // epochs

		t = epochs//5
		
		xticks = np.arange(0, len(history['lrs'])+1, t*batch_length)
		xlabels = np.arange(0, len(xticks))*t

		ax3.set_xticks(xticks, xlabels)
		
	else:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))
	
	epochs = np.arange(epochs)	


	ax1.plot(epochs, train_loss, label = 'train_loss')
	ax1.plot(epochs, val_loss, label = 'val_loss')

	ax2.plot(epochs, train_acc, label = 'train_acc')
	ax2.plot(epochs, val_acc, label = 'val_acc')

	ax1.legend()
	ax2.legend()

	ax1.set_xlabel('epochs')
	ax1.set_ylabel('loss')

	ax2.set_xlabel('epochs')
	ax2.set_ylabel('accuracy')


	plt.show()