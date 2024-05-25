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
		out = self.initial(X)
		out = self.Transposed(out)
		out = self.out(out)
		out = self.tanh(out)

		return out
