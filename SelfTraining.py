import torch
from utils import CustomDataSet
import torch.nn as nn

class SelfTraining: 
    def __init__(self, model:nn.Module))
def label_unk(model, mnist_labeled:CustomDataSet ,phase:int, device):
    model.eval()
    # minimum confidence for labelling, otherwise ignored
    # decreased non-linearly for each training phase
    print("Threshold: ",1-0.0005*(1+phase)**1.45)
    with torch.no_grad():
        for i, l in enumerate(mnist_labeled.data1):
            l=l.unsqueeze(0).to(device)
            outputs = model(l)
            ind=torch.argmax(outputs,axis=1).item()
            
            if mnist_labeled.labels1[i]==-1 :
                if outputs[0][ind].item() > max(0.55,1-0.0005*(1+phase)**1.45):
                    mnist_labeled.labels1[i]=ind
            else:
                # labels with less confidence are removed from traiing
                if outputs[0][ind].item() < 0.55:
                    mnist_labeled.labels1[i]=-1
    # modify training data (fn mentioned above)
    mnist_labeled.set_data_for_train()