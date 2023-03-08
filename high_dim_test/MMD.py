from scipy.stats import multivariate_normal as mvn
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import ghalton
import ot

## File Imports 
from sinkhorn_div import divergence
from get_plan import plan
from high_dim_tests import RankEnergy
from high_dim_tests import SoftRankEnergy
from high_dim_tests import two_sample_sinkdiv
from high_dim_tests import MaximumMeanDis_mix
from high_dim_tests import Wasserstein_1
from high_dim_tests import TwoSampleWTest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inDir = "/home/zimani/GenNets/energy_analysis/npy_files/"

n = 15000 

# tracks, showers, or mixed 
events = "tracks" 

# train or test
compare = "test"

outDir = "./npy_files/"
outFile = "MMD_"+events+"_"+compare+".npy"

larT = np.load(inDir+"larcv_png_64_"+compare+"_tracks.npy")
larS = np.load(inDir+"larcv_png_64_"+compare+"_showers.npy")

# Get selection of LArTPC events 
if events == 'tracks': 
	if n > larT.shape[0]: 
		n = larT.shape[0]
	lar = larT[0:n,:,:] 
if events == 'showers': 
	if n > larS.shape[0]: 
		n = larS.shape[0]
	lar = larS[0:n,:,:] 
if events == 'mixed': 
	if n > min(larT.shape[0], larS.shape[0]): 
		lar = np.concatenate((larT, larS)) 
		n = lar.shape[0]
	else: 
		larRatio = larS.shape[0]/larT.shape[0]
		if larRatio >= 1:
			print("Error: Bad Ratio")
			exit()
		larN = int(n*larRatio)
		lar = np.concatenate((larT[0:n-larN,:,:], larS[0:larN,:,:]))
lar = lar.flatten().reshape(n, 64*64)

print(n, "samples of event type", events) 

# Tunable hyperparameter for MMD 
#sigma_list= [1, 2, 4, 8, 16, 32]
sigma_list = []
for i in range(0,6): 
	sigma_list.append(2**(i+10)) 
#print(sigma_list) 

# Iterate all generated epochs
epochs = [20, 30, 40, 50, 100, 150, 300] 
MMDs = [] 
for epoch in epochs: 
	
	# Open generated epoch 
	genT = np.load(inDir+"gen_epoch"+str(epoch)+"_tracks.npy")
	genS = np.load(inDir+"gen_epoch"+str(epoch)+"_showers.npy")
	
	# Get selection of generated events  
	if events == 'tracks': 
		gen = genT[0:n,:,:]
	if events == 'showers': 
		gen = genS[0:n,:,:]
	if events == 'mixed': 
		genRatio = genS.shape[0]/genT.shape[0]
		if genRatio >= 1:
			print("Error: Bad Ratio")
			exit()
		genN = int(n*genRatio)
		gen = np.concatenate((genT[0:genN,:,:], genS[0:n-genN,:,:]))

	# Reshape generated data to 2D array 
	gen = gen.flatten().reshape(n, 64*64) 
	
	# Run high dimension analysis - from Shaoib
	outMMD = MaximumMeanDis_mix(lar, gen, sigma_list)
	#outSinkdiv = two_sample_sinkdiv(lar, gen, eps=1)
	#outW1 = Wasserstein_1(lar, gen)
	MMDs.append(outMMD.item()) 	

	print("Epoch", epoch, outMMD.item())

	# Clear GPU memory for next epoch -- NEED BATCHES TO SCALE 
	torch.cuda.empty_cache()
	del genT, genS, gen # Useless: Python does this automatically  


np.save(outDir+outFile, MMDs) 


