import torch
import torch.nn as nn
from torch import Tensor

import torch.nn as nn
from torch import Tensor, flatten

class Image_CLassification_Model(nn.Module):
	def __init__(self):
		super().__init__()
	

	def forward(self, image: Tensor):
		raise NotImplementedError

	@torch.no_grad()
	def evaluate(self, image: Tensor):
		self.eval()
		outs = self.forward(image)
		outs = torch.softmax(outs, dim=1)
		_, preds = torch.max(outs, dim=1)
		res = preds.sum().item()
		return res, outs[0][res].item()*100
	
	def load(self, file, device = 'cpu'):
		self.load_state_dict(torch.load(file, map_location=device))


class ConvBn(nn.Module):
	'''
	Convolution + BatchNorm + ReLu (+ MaxPool)

	keeping the size of input, if Maxpool, reduce the size by half
	'''
	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool=False) -> None:
		'''
		Convolution + BatchNorm + ReLu (+ MaxPool)

		keeping the size of input, if Maxpool, reduce the size by half
		'''
		super().__init__()
		self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		self.Bn = nn.BatchNorm2d(out_channels)
		self.act = nn.ReLU(inplace=True)
		
		if pool:
			self.pool = nn.MaxPool2d(2)
		else:
			self.pool = nn.Identity()

	def forward(self, X: Tensor):
		out = self.Conv(X)
		out = self.Bn(out)
		out = self.act(out)
		out = self.pool(out)
		return out

class ConvModel(nn.Module):
	'''
	return a Flatten with 512 features
	'''
	def __init__(self, in_channels, filters = [64, 128, 256, 512]):
		'''
		return a Flatten with 512 features
		'''
		super().__init__()

		self.initial = nn.Sequential(						# 3, n, n 		-> 64, n, n
			nn.Conv2d(in_channels, filters[0], 3, 1, 1),
			nn.ReLU(True)
		)

		layers = []

		for i in range(1, len(filters)):
			if i == len(filters)-1:
				layers.append(ConvBn(filters[i-1], filters[i]))
				break
			layers.append(ConvBn(filters[i-1], filters[i], pool=True))

		self.Conv = nn.Sequential(*layers)

		# self.adaptivePool = GlobalHybridPooling()
		self.adaptivePool = nn.AdaptiveMaxPool2d((1, 1))	# 512, n, n		-> 512, 1, 1

		self.flatten = nn.Flatten()

		
	def forward(self, X: Tensor):
		out = self.initial(X)
		out = self.Conv(out)

		out = self.adaptivePool(out)

		if out.dim() == 4:
			out = self.flatten(out)
		else:
			out = flatten(out)

		return out


class CNN(Image_CLassification_Model):
	def __init__(self, in_channels, n_classes):
		super().__init__()

		self.conv = ConvModel(in_channels)

		self.classifier = nn.Linear(512, n_classes)

	def forward(self, X: Tensor):
		if X.dim() == 3:
			X = X.unsqueeze(0)
		out = self.conv(X)
		out = self.classifier(out)
		return out


class GANSSL(Image_CLassification_Model):
	def __init__(self, in_channels, n_classes) -> None:
		super().__init__()

		self.conv = ConvModel(in_channels)

		self.dropout = nn.Dropout(0.5)

		self.classifier = nn.Linear(512, n_classes)
		
	def forward(self, X: Tensor):
		if X.dim() == 3:
			X = X.unsqueeze(0)

		out = self.conv(X)
		out = self.classifier(out)
		return out

import torch.nn as nn
from torch import Tensor

class TransposeBN(nn.Module):
	'''
	default: upsample, doubling input_size
	'''
	def __init__(self, in_channels, out_channels, kernel_size = 4, stride=2, padding=1, bias=False) -> None:
		'''
		default: upsample, doubling input_size
		'''
		super().__init__()

		self.deConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
		self.Bn = nn.BatchNorm2d(out_channels)
		self.act = nn.ReLU(inplace=True)
	
	def forward(self, X: Tensor):
		X = self.deConv(X)
		X = self.Bn(X)
		X = self.act(X)
		return X

class Generator(nn.Module):
	def __init__(self, latent_size, n_channels, filters = [512, 256, 128, 64]):
		super().__init__()

		self.initial = TransposeBN(latent_size, filters[0], 4, 1, 0)

		layers = []

		for i in range(1, len(filters)):
			layers.append(TransposeBN(filters[i-1], filters[i]))
		
		self.Transposed = nn.Sequential(*layers)

		self.out = nn.ConvTranspose2d(filters[-1], n_channels, 4, 2, 1, bias = False)

		self.tanh = nn.Tanh()

	def forward(self, X: Tensor):
		if X.dim() == 3:
			X = X.unsqueeze(0)
		out = self.initial(X)
		out = self.Transposed(out)
		out = self.out(out)
		out = self.tanh(out)

		return out
	
class Model:
	def __init__(self, train_ds, k=5):
		self.train_ds = train_ds
		self.k = k

	@staticmethod
	def euclidean(p1, p2):
		return torch.sqrt(torch.sum((p1-p2)**2, dim=[1,2,3]))
		
	
	def evaluate(self, test_point: torch.Tensor):
		distances = []
		

		X_train, y_train = self.train_ds[:]


		distances = self.euclidean(X_train, test_point).unsqueeze(1)

		distances = torch.cat((distances, y_train.unsqueeze(1)), dim=1)


		distances = distances[distances[:, 0].argsort()][: self.k]


		labels, counts = torch.unique(distances[:, 1], return_counts=True)

		majority_vote = labels[counts.argmax()]

		return majority_vote.item(), (counts.max() / self.k).item()


	def calculate_accuracy(self, test_ds):
		corrected = 0
	
		for test_point, label in test_ds:
			pred_label, _ = self.evaluate(test_point)
			corrected += (pred_label == label).item()
		
		return corrected / len(test_ds)
	