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

n = 100

# tracks, showers, or mixed 
events = "mixed" 

# train or test
compare = "test"

# Goodness of Fit test
# MMD, Sink, W1, 
GoF_test = "W2" 

outDir = "./npy_files/"
outFile = GoF_test+"_"+events+"_"+compare+".npy"

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
	larRatioT = larT.shape[0] / (larT.shape[0] + larS.shape[0]) 
	larN = int(n*larRatioT)
	lar = np.concatenate((larT[0:larN,:,:], larS[0:n-larN,:,:]))
	n = lar.shape[0]
lar = lar.flatten().reshape(n, 64*64)

print(n, "samples for", outFile) 

# Tunable hyperparameter for MMD 
#sigma_list= [1, 2, 4, 8, 16, 32]
sigma_list = []
for i in range(0,6): 
	sigma_list.append(2**(i+10)) 
#print(sigma_list)

# Iterate all generated epochs
epochs = [1, 10, 20, 30, 40, 50, 100, 150, 300] 
GoF = [] 
for i, epoch in enumerate(epochs): 
	
	# Open generated epoch 
	genT = np.load(inDir+"gen_epoch"+str(epoch)+"_tracks.npy")
	genS = np.load(inDir+"gen_epoch"+str(epoch)+"_showers.npy")

	# Get selection of generated events  
	if events == 'tracks': 
		gen = genT[0:n,:,:]
	if events == 'showers': 
		gen = genS[0:n,:,:]
	if events == 'mixed': 
		genRatioT = genT.shape[0] / (genT.shape[0] + genS.shape[0]) 
		genN = int(n*genRatioT)
		gen = np.concatenate((genT[0:genN,:,:], genS[0:n-genN,:,:]))

	# Reshape generated data to 2D array 
	gen = gen.flatten().reshape(n, 64*64) 
	
	if GoF_test == "MMD":  
		outMMD = MaximumMeanDis_mix(lar, gen, sigma_list)
		score = outMMD.item() 

	if GoF_test == "Sink":
		outSinkdiv = two_sample_sinkdiv(lar, gen, eps=1)
		score = outSinkdiv.item() 

	if GoF_test == "W1": 
		score = Wasserstein_1(lar, gen)

	if GoF_test == "RE": 
		score = RankEnergy(lar, gen) 

	if GoF_test == "SRE":
		score = SoftRankEnergy(lar, gen) 

	if GoF_test == "W2":
		score = TwoSampleWTest(lar, gen)[0]

	GoF.append(np.array([score, epoch, gen.shape[0]])) 	

	print("Epoch", epoch, score)

	# Clear GPU memory for next epoch -- NEED BATCHES TO SCALE 
	torch.cuda.empty_cache()
	del genT, genS, gen # Useless: Python does this automatically  

np.save(outDir+outFile, GoF) 


