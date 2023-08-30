import numpy as np
from tqdm import tqdm 
import torch
from scipy import linalg

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

larcv = np.load("larcv_png_64_train_FID.npy") 
#larcv = np.load("larcv_png_64_test_FID.npy") 
larcv = larcv.reshape(larcv.shape[0], -1) 

# Downsample larcv train to match larcv val 
if 1: 
	n = 10000 
	idxLar = np.random.choice(np.arange(larcv.shape[0]), size=n, replace=False) 
	larcv = larcv[idxLar] 

sigma1 = np.cov(larcv, rowvar=False) 
mu1 = larcv.mean(axis=0) 

fileNames = ['gen_epoch10', "gen_epoch20", "gen_epoch30", 
	"gen_epoch40", "gen_epoch50", "gen_epoch60", 
	"gen_epoch100", "gen_epoch150", "gen_epoch300", "VQVAE"] 


for fileName in fileNames:

	gen = np.load("./ssnet_activations/"+fileName+"_FID.npy") 
	gen = gen.reshape(gen.shape[0], -1) 
	
	# Downsample generated images to larcv comparison size 
	if larcv.shape[0] != gen.shape[0]: 
		idx = np.random.choice(np.arange(gen.shape[0]), size=larcv.shape[0], replace=False)  
		gen = gen[idx] 

	sigma2 = np.cov(gen, rowvar=False) 
	mu2 = gen.mean(axis=0) 

	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

	print(fileName, fid) 


exit() 

## GPU Code (WiP)  
#acts = torch.from_numpy(np.load(fileName+"_FID.npy")) 
#acts = acts.reshape(acts.size(0), -1) 

#sigma = torch.cov(acts.T) 
#sigmaNPY = sigma.cpu().detach().numpy()

#print(sigmaNPY.shape) 

#mu = torch.mean(acts, 0) 	
#muNPY = mu.cpu().detach().numpy() 

#print(muNPY.shape) 







