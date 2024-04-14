from torch import Tensor
import torch.nn as nn

class ConvModel(nn.Module):
	def __init__(self, inp_channel) -> None:
		super().__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(inp_channel, 32, kernel_size=(3, 3), stride=(2, 2), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)


		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)

		self.flatten = nn.Sequential(
			nn.AdaptiveAvgPool2d((2, 2)),
			nn.Flatten(),
		)


	def forward(self, X: Tensor):
		X = self.layer1(X)
		X = self.layer2(X)
		X = self.layer3(X)

		X = self.flatten(X)

		return X
	