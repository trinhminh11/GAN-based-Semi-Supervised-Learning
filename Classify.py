import torch
import torch.nn as nn
import torch.optim as optim
from Network_model import ConvModel
from utils import DeviceDataLoader
from torch import Tensor

import config
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
	def corrected(outputs, labels):
		_, preds = torch.max(outputs, dim=1)
		return torch.tensor(torch.sum(preds == labels).item())
	
	@torch.no_grad()
	def evaluate_with_loss(self, loader: DeviceDataLoader):
		self.eval()

		total_loss = 0
		total_corrected = 0

		for batch in loader:
			loss, corrected = self.step(batch)

			total_loss += loss.detach()
			total_corrected += corrected

		n = loader.num_data()

		return (total_loss/len(loader)).item(), (total_corrected/n).item()
	
	@torch.no_grad()
	def evaluate(self, loader: DeviceDataLoader):
		self.eval()

		total_corrected = 0

		for batch in loader:
			_, corrected = self.step(batch)

			total_corrected += corrected

		n = loader.num_data()

		return (total_corrected/n).item()
		
	@staticmethod
	def get_lr(optimizer: optim.Optimizer):
		for param_group in optimizer.param_groups:
			return param_group['lr']

	def step(self, batch) -> tuple[Tensor, Tensor]:
		images, labels = batch 
  
		outs = self(images)                  
		loss = self.criterion(outs, labels) 

		corrected = self.corrected(outs, labels)

		return loss, corrected


	def fit(self, epochs, max_lr, train_loader: DeviceDataLoader, val_loader: DeviceDataLoader, weight_decay=0, grad_clip=None, opt_func: optim.Optimizer = optim.SGD, threshold = 1, sched = True, PATH = "./", save_best = True):
		if config.DEVICE == 'cuda:0':
			torch.cuda.empty_cache()
		
		history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lrs': []}

		optimizer: optim.Optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)

		if sched:

			OneCycleLR = optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

		best_val = 0

		for epoch in range(epochs):
			self.train()
			train_loss = 0
			train_acc = 0
			lrs = []

			print(f"Epoch [{epoch}]")

			for batch in tqdm(train_loader):

				loss, corrected = self.step(batch)

				train_loss += loss.detach()
				
				train_acc += corrected

				loss.backward()

				if grad_clip:
					nn.utils.clip_grad_value_(self.parameters(), grad_clip)
				
				optimizer.step()
				optimizer.zero_grad()

				if sched:
					OneCycleLR.step()

				lrs.append(self.get_lr(optimizer))
			
			val_loss, val_acc = self.evaluate_with_loss(val_loader)
			train_loss = (train_loss/len(train_loader)).item()
			train_acc = (train_acc/train_loader.num_data()).item()


			

			history['val_loss'].append(val_loss)
			history['val_acc'].append(val_acc)
			history['train_loss'].append(train_loss)
			history['train_acc'].append(train_acc)
			history['lrs'] += lrs

			print(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}", end = "")

			if sched:
				print(f", lrs: {lrs[0]:.4f}->{lrs[-1]:.4f}", end = "")

			print()

			if val_acc >= best_val:
				if save_best:
					torch.save(self.state_dict(), PATH)
				
				best_val = val_acc

				if best_val > threshold:
					break
		
		return history
				