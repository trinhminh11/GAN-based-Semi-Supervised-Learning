from typing import overload
import os
import config


import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torchvision.datasets as datasets
import numpy as np

import torch.nn.functional as F
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt

from typing import Union


def get_PATH(name):
	'''
	get PATH to store data
	'''
	if config.NUM_LABELLED == -1:
		return f'{config.USED_DATA}/{name}/_full'
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
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class EMNIST(datasets.EMNIST):
	def __init__(self, root, train, split = "letters", download = False):
		super().__init__(root, split, train = train, download = download)


		self.data = torch.swapaxes(self.data, 1, 2)

		# self.data = tt.RandomHorizontalFlip(1)(self.data)

		self.targets -= 1

		self.classes = self.classes[1:]


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
	def __init__(self, x: Tensor, y: Tensor, transform = None) -> None:
		self.x: Tensor = x
		self.y: Tensor = y
		
		self.n_samples = len(x)

		self.transform = transform
	
	def to(self, device):
		self.x = self.x.to(device)
		self.y = self.y.to(device)


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

@overload
def supervised_samples(X: Tensor, y: Tensor, n_samples, n_classes, get_unsup = False) -> Union[tuple[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]: ...

@overload
def supervised_samples(ds: CustomDataSet, n_samples, n_classes, get_unsup = False) -> Union[tuple[CustomDataSet, CustomDataSet], CustomDataSet]: ...


def supervised_samples(a, b, c, d = False, e = False):
	if type(a) == CustomDataSet:
		ds = a
		X = ds.x
		y = ds.y
		n_samples = b
		n_classes = c
		get_unsup = d

	elif type(a) == Tensor and type(b) == Tensor:
		X = a
		y = b
		n_samples = c
		n_classes = d
		get_unsup = e
	
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
		if type(a) == CustomDataSet:
			return CustomDataSet(X_sup, y_sup, ds.transform), CustomDataSet(X_unsup, y_unsup, ds.transform)
		else:
			return X_sup, y_sup, X_unsup, y_unsup

	else:
		if type(a) == CustomDataSet:
			return CustomDataSet(X_sup, y_sup, ds.transform)
		else:
			return X_sup, y_sup


def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	
	return data.to(device, non_blocking=True)

def load_data(train_transform = None, test_transform = None):
	ldict = {}
	
	test_dataset: Dataset
	exec(f'train_dataset = {config.USED_DATA}(config.DATA_DIR, train = True)', globals().update(locals()), ldict)
	exec(f'test_dataset = {config.USED_DATA}(config.DATA_DIR, train = False)', globals().update(locals()), ldict)
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

	mean = [0.]*C
	std = [0.]*C

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

def cal_confusion_matrix(model, dl:DeviceDataLoader, n_classes = 10): 
	mat = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
	for batch in dl: 
		imgs, labels = batch
		labels = labels.to('cpu')
		pred = F.softmax(model(imgs).to('cpu') , dim = 1)
		for i in range(len(labels)): 
			mat[labels[i]][torch.argmax(pred[i])] += 1
	return np.array(mat)

def draw_cf_matrix(cf_matrix: np.ndarray, classes):
	# disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
	# 							display_labels=classes)

	df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
				  columns = [i for i in classes])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True, cmap='Blues')

	# disp.plot()
	# plt.show()

def precision(cf_matrix: np.ndarray): 
	p = []
	for i in range(cf_matrix.shape[0]): 
		TP = cf_matrix[i][i]
		T = cf_matrix.sum(axis = 0)[i]
		p.append(TP/T)
	return np.array(p)

def recall(cf_matrix: np.ndarray): 
	r = [] 
	for i in range(cf_matrix.shape[0]): 
		TP = cf_matrix[i][i]
		T = cf_matrix.sum(axis = 1)[i]
		r.append(TP/T)
	
	return np.array(r)

def f1_score(cf_matrix): 
	p = precision(cf_matrix)
	r = recall(cf_matrix)
	return np.divide(np.multiply(2*p, r), np.add(p, r))

def plotting(history: dict[str, dict[str, list[float]]]):
	epochs = history['epochs']
	num_axis = len(history)-1

	fig, axis = plt.subplots(1, num_axis, figsize = (8*num_axis, 5))

	try:
		axis[0]
	except:
		axis = [axis]

	idx = 0
	for name, y in history.items():
		if name == 'epochs':
			continue
		axis[idx].set_title(name)

		x = np.arange(len(y))
		axis[idx].plot(x, y, label = name)
		
		# if name == 'Learning rate':
		axis[idx].set_xticks(np.linspace(0, len(y), epochs)[::2], np.arange(0, epochs, 2))
		
		# else:
		# 	axis[idx].set_xticks(np.linspace(0, epochs, epochs)[::2], np.arange(0, epochs, 2))
		
		axis[idx].legend()
		axis[idx].set_xlabel('epochs')
		axis[idx].set_ylabel(name)
		idx += 1

	plt.show()
