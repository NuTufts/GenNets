import matplotlib.pyplot as plt
import numpy as np
import itertools 
import ot
from scipy.spatial import distance_matrix
import cv2 
import ot
from pyemd import emd 
from scipy.stats import wasserstein_distance



# From Shaoib
from high_dim_tests import MaximumMeanDis_mix as MMD
from sinkhorn_div import divergence as sinkDiv

# Mean Squared Error
def MSE(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

## Configurations 

inDir = "/home/zimani/GenNets/npy_files/" 

# "track" or "shower"
TorS = "shower" 

genFile = "gen_epoch50" 
gens = np.load(inDir+genFile+"_"+TorS+"s.npy") 

compFile = "larcv_png_64_train" 
comps = np.load(inDir+compFile+"_"+TorS+"s.npy") 

nNeighbors = 5 

# Choose which generated image to use -- select by eye 
genNum = 10 
gen = gens[genNum,:] 
if 0: 
	plt.imshow(gen, cmap='gray', interpolation='nearest')
	plt.show() 
	exit() 

# Hyperparameter for MMD 
sigma_list = []
for i in range(0,6): 
	sigma_list.append(2**(i+2)) 

# Edit image by rearrange quadrants
eRows = np.vstack([gen[31:], gen[:31]])
eCols = np.hstack([gen[:,31:], gen[:,:31]])
eBoth = np.hstack([eRows[:,31:], eRows[:,:31]])

# Save edited (rearranged) image 
if 0: 
	plt.imshow(genEdit, cmap='gray', interpolation='nearest')
	plt.tight_layout() 
	#plt.show() 
	plt.savefig("showerEdit.png")

# Good - Euclidian Norm
if 0: 
	print("l2") 
	l2Rows = int(np.linalg.norm(gen - eRows))
	l2Cols = int(np.linalg.norm(gen - eCols))
	l2Both = int(np.linalg.norm(gen - eBoth))
	print("Rows:", l2Rows) 
	print("Cols:", l2Cols) 
	print("Both:", l2Both)


# Bad - Empirical Distribution
if 0: 
	print("MMD") 
	mmdRows = round(MMD(gen, eRows, sigma_list).item(), 6)
	mmdCols = round(MMD(gen, eCols, sigma_list).item(), 6)
	mmdBoth = round(MMD(gen, eBoth, sigma_list).item(), 6)
	print("Rows:", mmdRows) 
	print("Cols:", mmdCols) 
	print("Both:", mmdBoth)

# Bad - Empircal Distribution 
if 0: 
	print("SinkDiv") 
	sdRows = round(sinkDiv(gen, eRows, eps=1).item(), 2)
	sdCols = round(sinkDiv(gen, eCols, eps=1).item(), 2)
	sdBoth = round(sinkDiv(gen, eBoth, eps=1).item(), 2)
	print("Rows:", sdRows) 
	print("Cols:", sdCols) 
	print("Both:", sdBoth) 

# Good? - Energy/Earth Mover Distance 
if 1:
	print("EMD") 

	# Get pairwise elements for 64x64 image
	#p = np.array(list(itertools.product(np.arange(64), repeat=2))) 

	# L2 distance matrix for all 
	#M = np.array(distance_matrix(p, p, p=2))

	xx, yy = np.meshgrid(np.arange(64), np.arange(64)) 
	xx = xx.ravel() 
	yy = yy.ravel() 

	# Image Histograms 
	genHist, _ = np.histogram(gen, bins=256, range=(0,255)) 
	genHist = genHist / np.sum(genHist) 
	eRowsHist, _ = np.histogram(eRows, bins=256, range=(0,255)) 
	eRowsHist = eRowsHist / np.sum(eRowsHist) 
	eColsHist, _ = np.histogram(eCols, bins=256, range=(0,255)) 
	eColsHist = eColsHist / np.sum(eColsHist) 
	eBothHist, _ = np.histogram(eBoth, bins=256, range=(0,255)) 
	eBothHist = eBothHist / np.sum(eBothHist) 

	# Not Working 
	#M = ot.dist(gen, eBoth) 
	#emdBoth = ot.emd2(gen, eBoth, M) 

	## Image Signature 
	genSig = np.vstack((gen.ravel()/np.sum(gen), xx, yy)).T.astype(np.float32)
	eRowsSig = np.vstack((eRows.ravel()/np.sum(eRows), xx, yy)).T.astype(np.float32)
	eColsSig = np.vstack((eCols.ravel()/np.sum(eCols), xx, yy)).T.astype(np.float32)
	eBothSig = np.vstack((eBoth.ravel()/np.sum(eBoth), xx, yy)).T.astype(np.float32)

	## L2 Distance matrix 
	p = np.array(list(itertools.product(np.arange(64), repeat=2))) 
	M = np.array(distance_matrix(p, p, p=2))

	## PyEMD 
	if 0: 
		emdSelf = emd(genHist, genHist, M) 
		emdRows = emd(eRowsHist, genHist, M) 
		emdCols = emd(eColsHist, genHist, M) 
		emdBoth = emd(eBothHist, genHist, M) 

	## Scipy.States EMD 
	if 0: 
		emdSelf = wasserstein_distance(genHist, genHist) 
		emdRows = wasserstein_distance(eRowsHist, genHist) 
		emdCols = wasserstein_distance(eColsHist, genHist) 
		emdBoth = wasserstein_distance(eBothHist, genHist) 

	## POT EMD 
	if 1: 
		emdSelf = ot.emd2(gen.ravel()/np.sum(gen), gen.ravel()/np.sum(gen), M) 
		emdRows = ot.emd2(gen.ravel()/np.sum(gen), eRows.ravel()/np.sum(eRows), M) 
		emdCols = ot.emd2(gen.ravel()/np.sum(gen), eCols.ravel()/np.sum(eCols), M) 
		emdBoth = ot.emd2(gen.ravel()/np.sum(gen), eBoth.ravel()/np.sum(eBoth), M) 

	if 0: 
		emdSelf, _, flow = cv2.EMD(genSig, genSig, cv2.DIST_L2) 
		emdRows, _, flow = cv2.EMD(eRowsSig, genSig, cv2.DIST_L2) 
		emdCols, _, flow = cv2.EMD(eColsSig, genSig, cv2.DIST_L2) 
		emdBoth, _, flow = cv2.EMD(eBothSig, genSig, cv2.DIST_L2) 
		
	print("Self:", emdSelf) 
	print("Rows:", emdRows) 
	print("Cols:", emdCols) 
	print("Both:", emdBoth)

# Save edited (flipped) image 
if 0: 
	plt.imshow(genEdit, cmap='gray', interpolation='nearest')
	plt.tight_layout() 
	#plt.show() 

	plt.savefig("showerEdit.png")

