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
	def __init__(self, in_channels):
		'''
		return a Flatten with 512 features
		'''
		super().__init__()
		
		self.conv1 = ConvBn(in_channels, 64)				# 3, 32, 32 		-> 64, n, n
		self.conv2 = ConvBn(64, 128, pool=True)				# 64, 32, 32 		-> 128, n//2, n//2
		self.conv3 = ConvBn(128, 256, pool=True)			# 128, 16, 16 		-> 256, n//4, n//4
		self.conv4 = ConvBn(256, 512)						# 256, 8, 8			-> 512, n//4, n//4
		# self.adaptivePool = GlobalHybridPooling()
		self.adaptivePool = nn.AdaptiveMaxPool2d((1, 1))	# 512, n//4, n//4 	-> 512, 1, 1

		self.flatten = nn.Flatten()

		
	def forward(self, X: Tensor):
		out = self.conv1(X)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)

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
	def __init__(self, in_channels, out_channels, kernel_size = 4, stride=2, padding=1) -> None:
		'''
		default: upsample, doubling input_size
		'''
		super().__init__()

		self.deConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
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
			nn.ConvTranspose2d(64, out_size[0], (3, 3), (2, 2)),
			nn.AdaptiveAvgPool2d((out_size[1], out_size[2])),
			nn.Tanh()
		)
	
	def forward(self, X: Tensor):
		X = self.deConv(X)

		X = self.out(X)

		return X

class Discriminator(nn.Module):
	def __init__(self, in_channels, n_classes) -> None:
		super().__init__()

		self.feature_extracter = ConvModel(in_channels)

		self.classifier = nn.Linear(512, n_classes)

		self.discriminator = nn.Linear(512, 1)
	
	def forward(self, X: Tensor):
		
		X = self.feature_extracter(X)

		return self.classifier(X), self.discriminator(X)
	
