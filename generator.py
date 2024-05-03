from torch import Tensor
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
	def __init__(self, inp_size, out_size) -> None:
		super().__init__()


		self.NN = nn.Sequential(
			nn.Linear(inp_size, 256*7*7),
			nn.LeakyReLU(0.2),
		)

		self.CONV = nn.Sequential(
			nn.ConvTranspose2d(256, 128, (3, 3), (2, 2)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(128, 64, (3, 3), (1, 1)),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			
		)

		self.out = nn.Sequential(
			nn.ConvTranspose2d(64, out_size[0], (3, 3), (2, 2)),
			nn.AdaptiveAvgPool2d((out_size[1], out_size[2])),
			nn.Tanh()
		)

		self.optimizer = optim.Adam(self.parameters(), lr = 0.0002, betas=[0.5, 0.999])
		self.criterion = nn.BCELoss()
	
	def forward(self, X: Tensor):
		X = self.NN(X)
		X = X.view(-1, 256, 7, 7)
		X = self.CONV(X)

		X = self.out(X)

		return X