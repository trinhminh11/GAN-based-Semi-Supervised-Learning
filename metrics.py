import numpy as np
from utils import DeviceDataLoader
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
def cal_confusion_matrix(model, dl:DeviceDataLoader, n_classes = 10): 
	mat = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
	for batch in dl: 
		imgs, labels = batch
		labels = labels.to('cpu')
		pred = F.softmax(model(imgs).to('cpu') , dim = 1)
		for i in range(len(labels)): 
			mat[labels[i]][torch.argmax(pred[i])] += 1
	return np.array(mat)

def draw_cf_matrix(cf_matrix: np.ndarray, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                display_labels=classes)
    disp.plot()
    plt.show()

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

def f1_score(precision: np.ndarray, recall: np.ndarray): 
	return np.divide(np.multiply(2*precision, recall), np.add(precision, recall))