import numpy as np
from utils import DeviceDataLoader
import torch
import torch.nn.functional as F
def cal_confusion_matrix(model, dl:DeviceDataLoader, n_classes = 10): 
	mat = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
	for batch in dl: 
		imgs, labels = batch
		labels = labels.to('cpu')
		pred = F.softmax(model(imgs).to('cpu') , dim = 1)
		for i in range(len(labels)): 
			mat[labels[i]][torch.argmax(pred[i])] += 1
	return np.array(mat)

def precision(cf_matrix: np.ndarray): 
	precision = []
	for i in range(cf_matrix.shape[0]): 
		TP = cf_matrix[i][i]
		T = cf_matrix.sum(axis = 0)[i]
		precision.append(TP/T)
	return np.array(precision)

def recall(cf_matrix: np.ndarray): 
	recall = [] 
	for i in range(cf_matrix.shape[0]): 
		TP = cf_matrix[i][i]
		T = cf_matrix.sum(axis = 1)[i]
		recall.append(TP/T)
	
	return np.array(recall)