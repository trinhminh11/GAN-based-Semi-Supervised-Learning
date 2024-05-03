import torch
import torch.nn as nn

class GlobalHybridPooling(nn.Module):
	def __init__(self, init_alpha = None):
		super().__init__()

		self.average = nn.AdaptiveAvgPool2d((1, 1))

		self.max = nn.AdaptiveMaxPool2d((1, 1))

		if init_alpha:
			self.alpha = nn.Parameter(torch.Tensor(init_alpha))
		else:
			self.alpha = nn.Parameter(torch.rand(1))
		
	def forward(self, X: torch.Tensor):
		avaragePool = self.average(X)
		maxPool = self.max(X)

		self.alpha.data = self.alpha.data.clamp(0, 1)

		return self.alpha * avaragePool + (1 - self.alpha) * maxPool


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

	def forward(self, X: torch.Tensor):
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

		
	def forward(self, X: torch.Tensor):
		out = self.conv1(X)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)

		out = self.adaptivePool(out)

		if out.dim() == 4:
			out = self.flatten(out)
		else:
			out = torch.flatten(out)

		return out

def main():
	c = ConvModel(3)

	X = torch.zeros(1, 3, 32, 32)
	out = c(X)
	print(out)


if __name__ == "__main__":
	main()

