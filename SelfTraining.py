import torch
from typing import Type
from utils import CreateDataLoader
from tqdm import tqdm
class SelfTraining: 
    def __init__(self, X_sup:torch.Tensor, y_sup:torch.Tensor, X_unsup:torch.Tensor, device): 
        self.X_sup = X_sup 
        self.y_sup = y_sup 
        self.X_unsup = X_unsup
        self.CEloss = torch.nn.CrossEntropyLoss()
        self.device = device 

    def selfTraining(self, epochs, teacher_model: torch.nn.Module, lr, optim: Type[torch.optim.Optimizer], transform):
         for epoch in range(epochs):
            teacher_model.train() 
            teacher_model.fit(self.X_sup, self.y_sup, lr, optim)
            teacher_model.eval() 
            pseudo_labels = teacher_model(self.X_unsup)
            student_model = teacher_model
            full_X = torch.cat(self.X_sup, self.X_unsup) 
            full_y = torch.cat(self.y_sup, pseudo_labels) 
            student_model.train() 
            student_model.fit(full_X, full_y, lr, optim)
            #  assign student model = teacher model 
            student_model = teacher_model