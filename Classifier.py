from typing import Type
import torch
import torch.nn as nn
import torch.optim as optim
from ConvModel import ConvModel
from utils import DeviceDataLoader
from torch import Tensor

from tqdm import tqdm


class Classifier(nn.Module):
	def __init__(self, in_channels, num_classes):
		super().__init__()

		self.conv = ConvModel(in_channels)

		self.classifier = nn.Linear(512, num_classes)

		self.criterion = nn.CrossEntropyLoss()
		
	def forward(self, X: Tensor):
		if X.dim() == 3:
			X = X.unsqueeze(0)
		out = self.conv(X)
		out = self.classifier(out)
		return out
	
	@staticmethod
	def corrected(preds, labels):
		return torch.tensor(torch.sum(preds == labels).item())
	
	@torch.no_grad()
	def evaluate(self, loader: DeviceDataLoader):
		self.eval()

		total_corrected = 0

		for batch in loader:
			images, labels = batch 
			outs = self(images)
			_, preds = torch.max(outs, dim=1)
			total_corrected += self.corrected(preds, labels)

		return (total_corrected/loader.num_data()).item() 
		

	def step(self, batch):
		images, labels = batch 
  
		outs = self(images)                  
		loss = self.criterion(outs, labels) 

		return loss


	def fit(self, epochs, max_lr, train_loader: DeviceDataLoader, opt_func: Type[optim.Optimizer] = optim.SGD, opt_params= {}, sched = True, PATH = "./", save = True):
		history: dict[str, list] = {'epochs': epochs, 'Loss': []}

		if sched:
			history['Learning rate'] = []

		optimizer: optim.Optimizer = opt_func(self.parameters(), max_lr, **opt_params) 

		if sched:
			OneCycleLR = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

		for epoch in range(epochs):
			self.train()
			lrs = []

			print(f"Epoch [{epoch}]")

			for batch in tqdm(train_loader):

				loss = self.step(batch)

				loss.backward()

				optimizer.step()
				optimizer.zero_grad()


				history['Loss'].append(loss.item())

				s = f"train_loss: {loss.item():.4f}"

				if sched:
					lrs.append(OneCycleLR.get_last_lr()[0])
					OneCycleLR.step()
					s += f", lrs: {lrs[0]:.4f}->{lrs[-1]:.4f}"

				tqdm.write(s, end="\r")
			
			tqdm.write("")

			if sched:
				history['Learning rate'] += lrs

			
			if save:
				torch.save(self.state_dict(), PATH)
		
		return history
				