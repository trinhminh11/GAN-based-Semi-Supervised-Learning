import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import config
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np

from typing import overload

def get_PATH(name):
	'''
	get PATH to store data
	'''
	if config.NUM_LABELLED == -1:
		return f'{config.USED_DATA}/{name}'
	else:
		return f'{config.USED_DATA}/{name}/_{config.NUM_LABELLED}'

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True

class CustomDataSet(Dataset):
	def __init__(self, x: Tensor, y: Tensor, transform = None) -> None:
		self.x: Tensor = x
		self.y: Tensor = y
		
		self.n_samples = len(x)

		self.transform = transform


	def __getitem__(self, index):
		x_sample = self.x[index]
		if self.y != None:
			y_sample = self.y[index]
		if self.transform:
			x_sample = self.transform(x_sample).type_as(x_sample)

		if self.y != None:
			return x_sample, y_sample
		else:
			return x_sample
	
	def __len__(self):
		return self.n_samples

class DeviceDataLoader():
	def __init__(self, dl: DataLoader, device):
		self.dl = dl
		self.device = device

		self.batch_size = self.dl.batch_size
		
	def __iter__(self):
		for b in self.dl: 
			yield to_device(b, self.device)

	def __len__(self):
		return len(self.dl)

	def num_data(self):
		return len(self.dl.dataset)

def one_hot(y):
	one_hot_y = torch.zeros((len(y), 10))
	one_hot_y[torch.arange(len(y)), y] = 1
	return one_hot_y

def supervised_samples(X: Tensor, y: Tensor, n_samples, n_classes, get_unsup = False):
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
		return X_sup, y_sup

def to_device(data: Tensor, device) -> Tensor:
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	
	return data.to(device, non_blocking=True)

def load_data(train_transform = None, test_transform = None):
	ldict = {}
	
	test_dataset: Dataset = None
	exec(f'train_dataset = datasets.{config.USED_DATA}(config.DATA_DIR, train = True, download = True, transform = train_transform)', globals().update(locals()), ldict)
	exec(f'test_dataset = datasets.{config.USED_DATA}(config.DATA_DIR, train = False, download = True, transform = test_transform)', globals().update(locals()), ldict)
	train_dataset: Dataset = ldict['train_dataset']
	test_dataset: Dataset = ldict['test_dataset']

	X_train = torch.Tensor(train_dataset.data).float()
	X_test = torch.Tensor(test_dataset.data).float()
	
	if X_train.dim() == 3:
		X_train = X_train.unsqueeze(1)
		X_test = X_test.unsqueeze(1)
	
	if X_train.shape[-1] == 3:
		X_train = X_train.permute(0, 3, 1, 2)
		X_test = X_test.permute(0, 3, 1, 2)
	
	Xmax = X_train.max()
	deno = 1

	if Xmax > 1:
		deno = 255
	
	X_train = X_train.div(deno)
	X_test = X_test.div(deno)
	
	y_train = torch.LongTensor(train_dataset.targets)
	y_test = torch.LongTensor(test_dataset.targets)

	return CustomDataSet(X_train, y_train, train_transform), CustomDataSet(X_test, y_test, test_transform), train_dataset.classes

	
	# return X_train, y_train, X_test, y_test, data.classes


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


@overload
def CreateDataLoader(X: Tensor, y: Tensor, *, batch_size = 1, transform = None, device = 'cpu') -> DeviceDataLoader: ...

@overload
def CreateDataLoader(dataset: Dataset, *, batch_size = 1, transform = None, device = 'cpu') -> DeviceDataLoader: ...


def CreateDataLoader(*args, batch_size = 1, transform = None, device = 'cpu'):
	if len(args) == 1:
		dataset = args[0]

	elif len(args) == 2:
		X, y = args
		dataset = CustomDataSet(X, y, transform)
	else:
		raise TypeError()
	
	dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=3, pin_memory=True)
	dl = DeviceDataLoader(dl, device)

	return dl


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