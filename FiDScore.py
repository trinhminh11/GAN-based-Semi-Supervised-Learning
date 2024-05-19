from utils import fid_score
import torch 
'''Inception3: Model calculate inception score'''
from torchvision.models import Inception3
import torchvision.transforms as tt

def fid_score(model: Inception3, used_data, real_imgs: torch.Tensor, target_imgs: torch.Tensor):
	'''scale imgs to fit the input size of Inception3.
	Expect input image of size (3, H, W). 
	Image have to be load in a range of [0, 1] and normalize by mean=[0.485, 0.456, 0.406] 
	and std = [0.229, 0.224, 0.225]'''
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	if used_data == 'MNIST':
		tfm = tt.Compose([
        tt.Grayscale(num_output_channels=3),
        tt.Resize(299),
		tt.Normalize(mean, std) 
     ])
	if used_data == 'CIFAR10': 
		tfm = tt.Compose([
			tt.Resize(299),
		    tt.Normalize(mean, std)
        ])
	real_imgs = tfm(real_imgs)
	target_imgs = tfm(target_imgs)
	# calculate targets
	real_labels = model(real_imgs) 
	target_labels = model(target_imgs) 
	# calculate mean and covariance of statistics
	m1, sigma1 = torch.mean(real_labels, dim = 0), torch.cov(real_labels)
	m2, sigma2 = torch.mean(target_labels, dim = 0), torch.cov(real_labels)
	# calculate sum square difference between means
	ssdiff = torch.sum((m1 - m2)**2)
	# calculate sqrt of product between cov 
	covmean = (torch.matmul(sigma1, sigma2))**0.5
	# calculate score 
	fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0*covmean) 
	return fid