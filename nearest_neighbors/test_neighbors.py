import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import ot

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
if 1: 
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

	#gen = gen.flatten() 
	#eRows = eRows.flatten()
	#eCols = eCols.flatten() 
	#eBoth = eBoth.flatten() 
	
	gen = gen / np.max(gen) 
	eRows = eRows / np.max(eRows)
	eCols = eCols / np.max(eCols) 
	eBoth = eBoth / np.max(eBoth) 
	
	# Not Working 
	M = ot.dist(gen, eBoth) 
	emdBoth = ot.emd2(gen, eBoth, M) 
	
	print("Both:", emdBoth) 

# Save edited (flipped) image 
if 0: 
	plt.imshow(genEdit, cmap='gray', interpolation='nearest')
	plt.tight_layout() 
	#plt.show() 

	plt.savefig("showerEdit.png")

