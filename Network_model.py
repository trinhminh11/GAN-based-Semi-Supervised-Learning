import torch.nn as nn
from torch import Tensor, flatten, argmax
import torch
import tqdm 
from typing import Type
import torch.optim as optim 
import torch.nn.functional as F
from utils import CustomDataSet, CreateDataLoader, ceil
import torchvision.transforms as tt


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
	
class Discriminator(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()

        self.Conv = ConvModel(in_channels)

        self.out = nn.Linear(512, n_classes+1)
        
    def forward(self, X: Tensor):
        out = self.Conv(X)
        out = self.out(out)
        return out
	
class GAN:
    def __init__(self, latent_size, n_channels, n_classes, device) -> None:
        self.generator = Generator(latent_size, n_channels)
        self.discriminator = Discriminator(n_channels, n_classes)

        self.latent_size = latent_size

        self.CEloss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCELoss()

        self.n_classes = n_classes

        self.resize = tt.Resize(32)

        self.device = device

        self.to(device)

    def to(self, device):
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
    
    def load_gen_state_dict(self, file):
        self.generator.load_state_dict(torch.load(file))
    
    def classifier_step(self, X, y):
        outs = self.discriminator(X)
        loss = self.CEloss(outs, y)

        return loss
    
    def discriminator_real_step(self, X):
        batch_size = X.shape[0]
        outs = self.discriminator(X)
        outs = F.softmax(outs, dim=1)[:, -1]   # shape: B x 1
        # outs is probability of discriminator predict fake images
        # because this is real images, we want this {outs} to be minimize
        
        y_hat = torch.zeros([batch_size], device=self.device)

        loss = self.BCEloss(outs, y_hat)
        
        return loss
        

    
    def discriminator_fake_step(self, batch_size):
        z = torch.randn([batch_size, self.latent_size, 1, 1], device = self.device)
        fake_images = self.generator(z)

        fake_images = self.resize(fake_images)

        outs = self.discriminator(fake_images)
        y_hat = torch.full([batch_size], self.n_classes, device=self.device)

        loss = self.CEloss(outs, y_hat)

        return loss


    
    def fit(self, epochs, batch_per_epoch, dis_lr, sup_ds: CustomDataSet, full_ds: CustomDataSet, optim: Type[optim.Optimizer], PATH = ".", save_best = False):
        optimizerD = optim(self.discriminator.parameters(), lr = dis_lr)

        sup_batch_size = ceil(len(sup_ds) / batch_per_epoch)
        full_batch_size = ceil(len(full_ds) / batch_per_epoch)

        sup_dataloader = CreateDataLoader(sup_ds, batch_size=sup_batch_size, device=self.device)
        full_dataloader = CreateDataLoader(full_ds, batch_size=full_batch_size, device = self.device)


        for epoch in range(epochs):

            for sup_b, full_b in tqdm(zip(sup_dataloader, full_dataloader), total = batch_per_epoch):
                sup_images, labels = sup_b
                full_images, _ = full_b

                _full_batch_size = full_images.shape[0]

                loss_classify = self.classifier_step(sup_images, labels)
                loss_real = self.discriminator_real_step(full_images)
                loss_fake = self.discriminator_fake_step(_full_batch_size)

                loss = loss_classify + loss_real + loss_fake

                loss.backward()

                optimizerD.step()

                tqdm.write(f'loss: {loss.detach().item()}', end = "\r")
    