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

inDir = "/home/zimani/GenNets/npy_files/"

n = 100 #Sink 
#n = 10000 #MMD 

# tracks, showers, or mixed 
events = "mixed" 

# train or test
compare = "train"

# Goodness of Fit test
# MMD, Sink, W1, 
GoF_test = "Sink" 

sinkEps = 0.1 

outDir = "./npy_files/"
if GoF_test == 'Sink': 
	outFile = GoF_test+"_"+events+"_"+compare+"_"+str(sinkEps)+".npy"
else: 
	outFile = GoF_test+"_"+events+"_"+compare+".npy"

larT = np.load(inDir+"larcv_png_64_"+compare+"_tracks.npy")
larS = np.load(inDir+"larcv_png_64_"+compare+"_showers.npy")


# Get selection of LArTPC events 
if events == 'tracks': 
	lars = larT
if events == 'showers': 
	lars = larS
if events == 'mixed': 
	lars = np.concatenate((larT, larS)) 
if n > lars.shape[0]: 
	n = lars.shape[0]
idxL = np.random.choice(np.arange(lars.shape[0]), size=n, replace=False) 
lar = lars[idxL]
lar = lar.flatten().reshape(n, 64*64)

print(n, "samples for", outFile) 

# Tunable hyperparameter for MMD 
#sigma_list= [1, 2, 4, 8, 16, 32]
sigma_list = []
for i in range(0,6): 
	sigma_list.append(2**(i+10)) 
#print(sigma_list)

#np.random.sample(lars, n) 
#np.random.sample(gens, n) 

# Iterate all generated epochs
epochs = [1, 5, 10, 20, 30, 40, 50, 60, 100, 150, 300] 
GoF = [] 
for i, epoch in enumerate(epochs): 
	
	# Open generated epoch 
	genT = np.load(inDir+"gen_epoch"+str(epoch)+"_tracks.npy")
	genS = np.load(inDir+"gen_epoch"+str(epoch)+"_showers.npy")

	# Get selection of generated events  
	if events == 'tracks': 
		gens = genT 
	if events == 'showers': 
		gens = genS
	if events == 'mixed': 
		gens = np.concatenate((genT, genS)) 
	if n > gens.shape[0]: 
		print("Error: n > gens.shape[0]") 
		exit()  
	idxG = np.random.choice(np.arange(gens.shape[0]), size=n, replace=False) 
	gen = gens[idxG] 

	# Reshape generated data to 2D array 
	gen = gen.flatten().reshape(n, 64*64) 
	
	if GoF_test == "MMD":  
		outMMD = MaximumMeanDis_mix(lar, gen, sigma_list)
		score = outMMD.item() * np.sqrt(n)

	if GoF_test == "Sink":
		outSinkdiv = two_sample_sinkdiv(lar, gen, eps=sinkEps)
		score = outSinkdiv.item() 

	if GoF_test == "W1": 
		score = Wasserstein_1(lar, gen)

	if GoF_test == "RE": # Not work for single image
		score = RankEnergy(lar, gen) 

	if GoF_test == "SRE": # Not work for single image 
		score = SoftRankEnergy(lar, gen) 

	if GoF_test == "W2": # Slow for 100 images 
		score = TwoSampleWTest(lar, gen)[0]

	GoF.append(np.array([score, epoch, gen.shape[0]])) 	

	print("Epoch", epoch, score)

	# Clear GPU memory for next epoch -- NEED BATCHES TO SCALE 
	torch.cuda.empty_cache()
	del genT, genS, gen # Useless: Python does this automatically  

np.save(outDir+outFile, GoF) 


