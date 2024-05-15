import torch.nn as nn
from torch import Tensor, flatten, argmax


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
	def __init__(self, inp_size, out_size) -> None:
		super().__init__()

		out_channels = out_size[0]

		ngf = 128

		# inp_sizex1x1
		self.deConv = nn.Sequential(
			TransposeBN(inp_size, ngf*8, 4, 1, 0),	# 1024x4x4
			TransposeBN(ngf*8, ngf*4),				# 512x8x8
			TransposeBN(ngf*4, ngf*2),				# 256x16x16
			TransposeBN(ngf*2, ngf),				# 128x32x32
		)

		self.out = nn.Sequential(
			nn.ConvTranspose2d(ngf, out_size[0], (3, 3), (2, 2)),
			nn.AdaptiveAvgPool2d((out_size[1], out_size[2])),
			nn.Tanh()
		)
	
	def forward(self, X: Tensor):
		if X.dim() == 1:
			size = len(X)
			X = X.reshape(size, 1, 1).unsqueeze(0)
		
		elif X.dim() == 2:
			B, size = X.shape
			X = X.reshape(B, size, 1, 1)

		X = self.deConv(X)

		X = self.out(X)

		return X

class Discriminator(nn.Module):
	def __init__(self, in_channels, n_classes) -> None:
		super().__init__()

		self.n_classes = n_classes

		self.feature_extracter = ConvModel(in_channels)

		self.discriminator = nn.Linear(512, n_classes+1)
	
	def _forward_imply(self, X: Tensor):
		if X.dim() == 2:
			X = X.unsqueeze(0)

		if X.dim() == 3:
			X = X.unsqueeze(0)
		
		X = self.feature_extracter(X)

		return X
	
	def forward(self, X: Tensor):
		X = self._forward_imply(X)

		return self.discriminator(X)
	
	def classify(self, X: Tensor):
		return self.forward(X)[:, :-1]
	
	def discriminate(self, X):
		return self.forward(X)[:, -1]
		

