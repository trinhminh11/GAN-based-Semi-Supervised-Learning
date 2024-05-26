import torch.nn as nn
from torch import Tensor, flatten


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
