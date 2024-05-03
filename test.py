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

		self.alpha = torch.clamp(self.alpha, 0, 1)

		return self.alpha * avaragePool + (1 - self.alpha) * maxPool

g = GlobalHybridPooling()