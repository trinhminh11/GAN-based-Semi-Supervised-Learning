import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as tt
import config

def load_data(train_transform = None, test_transform = None):
	ldict = {}
	
	test_dataset: Dataset = None
	print(f'train_dataset = datasets.{config.USED_DATA}(config.DATA_DIR, train = True, download = True, transform = train_transform)')
	exec(f'train_dataset = datasets.{config.USED_DATA}(config.DATA_DIR, train = True, download = True, transform = train_transform)', globals().update(locals()), ldict)
	exec(f'test_dataset = datasets.{config.USED_DATA}(config.DATA_DIR, train = False, download = True, transform = test_transform)', globals().update(locals()), ldict)
	train_dataset: Dataset = ldict['train_dataset']
	test_dataset: Dataset = ldict['test_dataset']

	X_train = torch.Tensor(train_dataset.data)
	X_test = torch.Tensor(test_dataset.data)
	if X_train.dim() == 3:
		X_train = X_train.unsqueeze(1)
		X_test = X_test.unsqueeze(1)
	
	if X_train.shape[-1] == 3:
		X_train = X_train.permute(0, 3, 1, 2)
		X_test = X_test.permute(0, 3, 1, 2)
	
	X_train = tt.RandAugment().forward(X_train)
	return X_train, X_test

X_train, X_test = load_data() 
X_train.to(torch.uint8)
print(X_train[0])