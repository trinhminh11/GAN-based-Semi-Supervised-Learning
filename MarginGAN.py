from Network_model import Generator, Discriminator
import torch
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim
from utils import CustomDataSet 
from typing import Type
from Network_model import ConvModel
from torch.utils.data import random_split

from tqdm import tqdm

class Classifier(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()

        self.Conv = ConvModel(in_channels)

        self.dropout = nn.Dropout(0.5)

        self.out = nn.Linear(512, n_classes)
        
    def forward(self, X: Tensor):
        out = self.Conv(X)
        out = self.dropout(out)
        out = self.out(out) 
        return out
    
class MarginGAN: 
    def __init__(self, latent_size, n_channels, n_classes, device): 
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.generator = Generator(latent_size, n_channels)
        self.discriminator = Discriminator(n_channels, 1) 
        self.classifier = Classifier(n_channels, n_classes)

        self.CEloss = nn.CrossEntropyLoss() 
        self.BCEloss = nn.BCELoss()
        self.device = device 
        self.to(device)
    
    def to(self, device): 
        self.generator.to(device) 
        self.discriminator.to(device)
        self.classifier.to(device)
    
    def load_gen_state_dict(self, file):
        self.generator.load_state_dict(torch.load(file))
    
    def accuracy(self, test_dl): 
        corrected = 0
        for b in tqdm(test_dl):
            images, y = b
            outs = self.classifier.forward(images)
            outs = torch.argmax(outs[:, :-1], dim=1)
            corrected += (outs == y).sum().item()
        return corrected / test_dl.num_data()
    
    def discriminator_step(self, real_imgs:torch.Tensor, batch_size): 
        # Train discriminator to recognize real imgs as real imgs
        real_batch_size = real_imgs.shape[0]
        outs = self.discriminator(real_imgs)
        outs = F.softmax(outs, dim = 1)[:, 0]
        # outs is probability of discriminator predict fake images
        # because this is real images, we want this {outs} to be minimize
        y_hat = torch.zeros([real_batch_size], device = self.device) 
        real_loss = self.BCEloss(outs, y_hat)

        # Train discriminator to recognize fake imgs as fake imgs 
        z = torch.randn([batch_size, self.latent_size, 1, 1], device = self.device)
        fake_imgs = self.generator(z)
        fake_outs = self.discriminator(fake_imgs)
        fake_outs = F.softmax(fake_outs, dim = 1)[:, 1]
        # outs is probability of discriminator predict fake images
        # because this is fake images, we want this {outs} to be maximize
        fake_y_hat = torch.ones([batch_size], device = self.device) 
        fake_loss = self.BCEloss(fake_outs, fake_y_hat)
        return real_loss + fake_loss
    
    def classifier_step(self, sup_imgs, sup_labels, unsup_imgs, batch_size): 
        # Loss for labeled samples
        sup_outs = self.classifier(sup_imgs) 
        sup_loss = self.CEloss(sup_outs, sup_labels)

        # Loss for unlabeled samples
        # Pseudo_label:  Pick up the class which
        # has maximum predicted probability for each unlabeled
        # sample
        unsup_outs = self.classifier(unsup_imgs) 
        unsup_pseudolabels = torch.argmax(unsup_outs, dim = 1) 
        # print(unsup_pseudolabels.shape)
        unsup_loss = self.CEloss(unsup_outs, unsup_pseudolabels)

        # Loss for generated samples. Also pseudo_labelling as for 
        # unsup imgs, but now apply the inverted binary cross entropy 
        # as loss. Aim: decrease the margin of these data points
        # and make the prediction distribution flat
        z = torch.randn([batch_size, self.latent_size, 1, 1], device = self.device)
        fake_imgs = self.generator(z) 
        fake_outs = self.classifier(fake_imgs)
        fake_pseudolabels = torch.argmin(fake_outs, dim = 1) 
        fake_loss = self.CEloss(fake_outs, fake_pseudolabels) 

        return sup_loss + unsup_loss + fake_loss

    def fit(self, epochs, batch_size, batch_per_epoch, dis_lr, max_lr, sup_ds:CustomDataSet, unsup_ds:CustomDataSet, full_ds:CustomDataSet, test_dl, optim:Type[optim.Optimizer], weight_decay = 0, sched = True, PATH = ".", save_best = False, grad_clip = False): 
        optimizerD = optim(self.discriminator.parameters(), lr = dis_lr)
        optimizerC = optim(self.classifier.parameters(), lr = max_lr, weight_decay = weight_decay)

        if sched: 
            OneCycleLR = torch.optim.lr_scheduler.OneCycleLR(optimizerC, max_lr, epochs=epochs, steps_per_epoch=batch_per_epoch)

        self.discriminator.train()
        self.classifier.train() 
        for epoch in (range(epochs)):
            for i in range(batch_per_epoch): 
                sup_imgs, labels = random_split(sup_ds, [batch_size, len(sup_ds) - batch_size])[0][:]
                full_imgs = random_split(full_ds, [batch_size, len(full_ds) - batch_size])[0][:]
                unsup_imgs = random_split(unsup_ds, [batch_size, len(unsup_ds) - batch_size])[0][:]
                # train discriminator
                if (i%2 == 1):
                    D_loss = self.discriminator_step(full_imgs.to(self.device), batch_size)
                    D_loss.backward()
                    optimizerD.step()
                    tqdm.write(f'D_loss: {D_loss.detach().item()}', end = "\r")
                # train classifier
                C_loss = self.classifier_step(sup_imgs.to(self.device), labels.to(self.device), unsup_imgs.to(self.device), batch_size) 
                C_loss.backward()
                
                if grad_clip: 
                    torch.nn.utils.clip_grad_value_(self.classifier.parameters(), 0.1)
                
                optimizerC.step()
                optimizerC.zero_grad()
                if sched: 
                    OneCycleLR.step()
                
                tqdm.write(f'C_loss: {C_loss.detach().item()}', end = "\r")
                
            self.discriminator.eval() 
            self.classifier.eval()
            tqdm.write(f'accuracy: {self.accuracy(test_dl)}', end = "\r")