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

genName = "gen_epoch100"
genT = np.load(inDir+genName+"_tracks.npy")
genS = np.load(inDir+genName+"_showers.npy")
genRatio = genS.shape[0]/genT.shape[0]
if genRatio >= 1:
	print("Error: Bad Ratio")
	exit()
genN = int(n*genRatio)
gen = np.concatenate((genT[0:genN,:,:], genS[0:n-genN,:,:]))


larT = np.load(inDir+"larcv_png_64_train_tracks.npy")
larS = np.load(inDir+"larcv_png_64_train_showers.npy")
larRatio = larS.shape[0]/larT.shape[0]
if larRatio >= 1:
	print("Error: Bad Ratio")
	exit()
larN = int(n*larRatio)
lar = np.concatenate((larT[0:larN,:,:], larS[0:n-larN,:,:]))


#lar = np.load("./samples/larcv_png_64_train_tracks_5k.npy")
#gen = np.load("./samples/gen_epoch100_tracks_5k.npy")

#n = min(lar.shape[0]+lar.shape[0], genT.shape[0]+genS.shape[0])

#lar  = larT[0:n,:,:]
#gen = genT[0:n,:,:]


if gen.shape[0] == lar.shape[0]:
	print(genName, lar.shape[0], "samples")
else:
	print("Error: Different Shapes")
	exit()


lar = lar.flatten().reshape(n, 64*64)
gen = gen.flatten().reshape(n, 64*64)

sigma_list= [1, 2, 4, 8, 16, 32] # tunable hyperparameter for MMD

outRE = '' 
outSRE = '' 
outSinkdiv = '' 
outMMD = '' 
outW1 = '' 
outWQT = '' 

#outRE = RankEnergy(lar, gen)
#outSRE = SoftRankEnergy(lar, gen, eps=1)
outSinkdiv = two_sample_sinkdiv(lar, gen, eps=1)
outMMD = MaximumMeanDis_mix(lar, gen, sigma_list)
outW1 = Wasserstein_1(lar, gen)
#outWQT = TwoSampleWTest(lar, gen)

print("Rank Energy:", outRE)
print("Soft Rank Energy:", outSRE)
print("MMD:", outMMD.item())
print("Sinkhorn Divergence:", outSinkdiv.item())
print("Wasserstein-1:", outW1)
print("Wasserstein-Quantile distance:", outWQT)
