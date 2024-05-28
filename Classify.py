import torch.nn as nn 
from ConvModel import ConvModel
from torch import Tensor
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