import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.grad
import torch.optim as optim
import torch.utils
from torch.utils.data import random_split
import torchvision.transforms as tt
from utils import DeviceDataLoader, CustomDataSet
from ConvModel import ConvModel
from copy import deepcopy


from tqdm.notebook import tqdm

from typing import Type


def custom_function(X: Tensor):
	Z_x = torch.sum(torch.exp(X), dim=-1)
	D_x = Z_x / (Z_x+1)
	return D_x

class Discriminator(nn.Module):
	def __init__(self, in_channels, n_classes) -> None:
		super().__init__()

		self.conv = ConvModel(in_channels)

		self.dropout = nn.Dropout(0.5)

		self.classifier = nn.Linear(512, n_classes)
		
	def forward(self, X: Tensor):
		out = self.conv(X)
		out = self.dropout(out)
		out = self.classifier(out)
		return out

class GANSSL:
	def __init__(self, generator: nn.Module, discriminator: nn.Module, latent_size, device) -> None:
		self.generator = generator
		self.discriminator = discriminator
		# self.generator = Generator(latent_size, n_channels)
		# self.discriminator = Discriminator(n_channels, n_classes)

		self.latent_size = latent_size

		self.CEloss = nn.CrossEntropyLoss()
		self.BCEloss = nn.BCELoss()

		self.resize = tt.Resize(32)

		self.device = device

		self.to(device)

	def to(self, device):
		self.generator = self.generator.to(device, non_blocking=True)
		self.discriminator = self.discriminator.to(device, non_blocking=True)
	
	def load_dis_state_dict(self, file):
		self.discriminator.load_state_dict(torch.load(file))
	
	def load_gen_state_dict(self, file):
		self.generator.load_state_dict(torch.load(file))
	
	@torch.no_grad()
	def evaluate(self, dataloader: DeviceDataLoader):
		self.discriminator.eval()
		corrected = 0
	
		for b in dataloader:
			images, labels = b
			outs = self.discriminator.forward(images)
			_, preds = torch.max(outs, dim=1)
			outs = torch.argmax(outs, dim=1)
			corrected += torch.sum(preds == labels).item()
		
		return corrected / dataloader.num_data()

	def classifier_step(self, X, y):
		outs = self.discriminator(X)
		loss = self.CEloss(outs, y)

		return loss

	def discriminator_step(self, X: Tensor, y: Tensor):
		outs = self.discriminator(X)
		outs = custom_function(outs)
		loss = self.BCEloss(outs, y)

		return loss
	
	def discriminator_real_step(self, X):
		batch_size = X.shape[0]
		
		y_hat = torch.ones([batch_size], device = self.device)

		loss = self.discriminator_step(X, y_hat)
		
		return loss

	
	def discriminator_fake_step(self, batch_size):
		z = torch.randn([batch_size, self.latent_size, 1, 1], device=self.device)
		fake_images = self.generator(z)

		fake_images = self.resize(fake_images)

		y_hat = torch.zeros([batch_size], device=self.device)

		loss = self.discriminator_step(fake_images, y_hat)

		return loss
	
	def generator_step(self, batch_size):
		z = torch.randn([batch_size, self.latent_size, 1, 1], device=self.device)
		fake_images = self.generator(z)

		fake_images = self.resize(fake_images)

		outs = self.discriminator(fake_images)

		outs = torch.softmax(outs, dim=1)[:, -1]

		y_hat = torch.zeros([batch_size], device=self.device)

		loss = self.BCEloss(outs, y_hat)

		return loss
	
	
	def fit(self, epochs, batch_size, batch_per_epoch, dis_lr, sup_ds: CustomDataSet, full_ds: CustomDataSet, test_dl, optimizer: Type[optim.Optimizer], opt_params = {}, sched = False, PATH = ".", save = False, keep_best = True):

		history: dict[str, list] = {'epochs': epochs, 'Loss': []}

		if sched:
			history['Learning rate'] = []

		optimizerD = optimizer(self.discriminator.parameters(), lr = dis_lr, **opt_params)

		# optimizerG = optimizer(self.generator.parameters(), lr = dis_lr, **opt_params)

		if sched:
			OneCycleLR = optim.lr_scheduler.OneCycleLR(optimizerD, dis_lr, epochs*batch_per_epoch)

		n_sup = len(sup_ds)
		n_data = len(full_ds)

		best_acc = -1

		best_model = None


		for epoch in range(epochs):
			self.discriminator.train()

			lrs = []

			print(f"Epoch [{epoch}]:")
			for i in tqdm(range(batch_per_epoch)):
				sup_images, labels = random_split(sup_ds, [batch_size, n_sup-batch_size])[0][:]
				C_loss = self.classifier_step(sup_images.to(self.device), labels.to(self.device)) 


				unsup_images, _ = random_split(full_ds, [batch_size, n_data-batch_size])[0][:]
				real_loss = self.discriminator_real_step(unsup_images.to(self.device))
				fake_loss = self.discriminator_fake_step(batch_size)
				D_loss = (real_loss + fake_loss)/10

				loss = C_loss + D_loss
				loss.backward()

				# torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)

				# torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), 0.1)

				optimizerD.step()


				# G_loss = self.generator_step(batch_size)
				# G_loss.backward()
				# optimizerG.step()

				history['Loss'].append(C_loss.item())

				s = f'C_Loss: {C_loss:.5f}, D_Loss: {D_loss:.5f}'
				
				if sched:
					lrs.append(OneCycleLR.get_last_lr()[0])
					OneCycleLR.step()
					s += f', lrs: {lrs[0]:.6f}-> {lrs[-1]:.6f}'

				tqdm.write(s, end = "\r")
			
			acc = self.evaluate(test_dl)

			if acc > best_acc:
				best_acc = acc
				best_model = deepcopy(self.discriminator.state_dict())
			
			tqdm.write("")

			tqdm.write(f"acc = {self.evaluate(test_dl)}")

			if sched:
				history['Learning rate'] += lrs

		
		if keep_best:
			self.discriminator.load_state_dict(best_model)

		if save:
			torch.save(self.discriminator.state_dict(), PATH)
				
				
		return history