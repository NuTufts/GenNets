import numpy as np
from scipy import linalg

#import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

larcvT = np.load("./ssnet_activations/larcv_png_64_train_FID.npy") 
larcvT = larcvT.reshape(larcvT.shape[0], -1) 
larcvV = np.load("./ssnet_activations/larcv_png_64_test_FID.npy") 
larcvV = larcvV.reshape(larcvV.shape[0], -1) 

# Downsample training larcv dataset to larcv dataset size   
if larcvT.shape[0] > larcvV.shape[0]: 
	idxLar = np.random.choice(np.arange(larcvT.shape[0]), size=larcvV.shape[0], replace=False) 
	larcvT = larcvT[idxLar] 

sigmaT = np.cov(larcvT, rowvar=False) 
muT = larcvT.mean(axis=0) 
sigmaV = np.cov(larcvV, rowvar=False) 
muV = larcvV.mean(axis=0) 

fileNames = ["gen_epoch10", "gen_epoch20", "gen_epoch30", 
	"gen_epoch40", "gen_epoch50", "gen_epoch60", 
	"gen_epoch100", "gen_epoch150", "gen_epoch300", "VQVAE"] 

FIDs = [] 

print("FileName, Training FID, Validation FID") 

for i, fileName in enumerate(fileNames):

	gen = np.load("./ssnet_activations/"+fileName+"_FID.npy") 
	gen = gen.reshape(gen.shape[0], -1) 
	
	# Downsample generated images to larcv comparison size 
	if larcvV.shape[0] != gen.shape[0]: 
		idx = np.random.choice(np.arange(gen.shape[0]), size=larcvV.shape[0], replace=False)  
		gen = gen[idx] 

	sigma = np.cov(gen, rowvar=False) 
	mu = gen.mean(axis=0) 
	
	# Training FID 
	# calculate sum squared difference between means
	ssdiffT = np.sum((muT - mu)**2.0)
	# calculate sqrt of product between cov
	covmeanT = linalg.sqrtm(sigmaT.dot(sigma))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmeanT):
		covmeanT = covmeanT.real
	# calculate score
	fidT = ssdiffT + np.trace(sigmaT + sigma - 2.0 * covmeanT)

	# Repeate for validation FID
	ssdiffV = np.sum((muV - mu)**2.0)
	covmeanV = linalg.sqrtm(sigmaV.dot(sigma))
	if np.iscomplexobj(covmeanV):
		covmeanV = covmeanV.real
	fidV = ssdiffV + np.trace(sigmaV + sigma - 2.0 * covmeanV)

	print(fileName, fidT, fidV) 
	FIDs.append(fileName, fidT, fidV) 

np.save("FID_values.npy", np.array(FIDs))  


## GPU Code (WiP) -- CPU is sufficient for now 

#acts = torch.from_numpy(np.load(fileName+"_FID.npy")) 
#acts = acts.reshape(acts.size(0), -1) 

#sigma = torch.cov(acts.T) 
#sigmaNPY = sigma.cpu().detach().numpy()

#print(sigmaNPY.shape) 

#mu = torch.mean(acts, 0) 	
#muNPY = mu.cpu().detach().numpy() 

#print(muNPY.shape) 







