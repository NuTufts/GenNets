import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np

# From Shaoib 
from high_dim_tests import MaximumMeanDis_mix as MMD
from sinkhorn_div import divergence as SinkDiv 

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

# l2 or MMD or SinkDiv 
distMode = "SinkDiv" 

# Choose which generated image to use -- select by eye 
genNum = 10 
gen = gens[genNum,:] 
if 0: 
	plt.imshow(gen, cmap='gray', interpolation='nearest')
	plt.show() 
	exit() 

# Optional: apply small amount of gaussian noise to the image
#gen = gaussian_filter(gen, sigma=1)

# Hyperparameter for MMD 
sigma_list = []
for i in range(0,6): 
	sigma_list.append(2**(i+2)) 

# Hyperparameter for SinkDiv
eps = 1 

## End Configuration 

neighbors = [] 
dists = [] 

for comp in comps: 

	if distMode == 'l2': 
		dist = int(np.linalg.norm(gen - comp))
	elif distMode == 'MMD': 
		dist = round(MMD(gen, comp, sigma_list).item(),6) 
	elif distMode == "SinkDiv": 
		dist = round(SinkDiv(gen, comp, eps).item(), 2)
	else: 
		print("Error: invalid distance mode") 
		exit() 

	# First n elements go in the list 
	if len(dists) < nNeighbors: 
		dists.append(dist) 
		neighbors.append(comp) 
		continue 

	# Replace largest distance with smaller distance 
	if dist < max(dists): 
		maxLoc = dists.index(max(dists)) 
		dists.pop(maxLoc) 
		neighbors.pop(maxLoc) 
		dists.append(dist) 
		neighbors.append(comp) 

dists = np.asarray(dists) 
neighbors = np.asarray(neighbors) 

# Sort nearest neighbors 
distSort = (-1*dists).argsort() 
dists = dists[distSort[::-1]] 
neighbors = neighbors[distSort[::-1]] 

## Repeat with generated datset (self)
neighbors1 = [] 
dists1 = [] 

for comp1 in gens: 
	
	if distMode == 'l2': 
		dist1 = int(np.linalg.norm(gen - comp1))
	if distMode == 'MMD': 
		dist1 = round(MMD(gen, comp1, sigma_list).item(), 6)
	if distMode == "SinkDiv": 
		dist1 = round(SinkDiv(gen, comp1, eps).item(), 2)

	# First n elements go in the list 
	if len(dists1) < nNeighbors+1: 
		dists1.append(dist1) 
		neighbors1.append(comp1) 
		continue 

	# Replace largest distance with smaller distance 
	if dist1 < max(dists1): 
		maxLoc = dists1.index(max(dists1)) 
		dists1.pop(maxLoc) 
		neighbors1.pop(maxLoc) 
		dists1.append(dist1) 
		neighbors1.append(comp1) 

dists1 = np.asarray(dists1) 
neighbors1 = np.asarray(neighbors1) 

# Sort nearest neighbors 
distSort1 = (-1*dists1).argsort() 
dists1 = dists1[distSort1[::-1]] 
neighbors1 = neighbors1[distSort1[::-1]] 

# Remove self match - need adjust nNeighbors
#neighbors1 = neighbors1[1:]


## Plotting 

plt.figure(figsize=(14,4)) 

plt.subplot2grid(shape=(2, 7), loc=(0,0), colspan=2, rowspan=2) 
plt.imshow(gen, cmap='gray', interpolation='nearest') 
plt.title("Generated") 
plt.axis('off') 

for i in range(nNeighbors): 
	plt.subplot2grid((2,7), (0,i+2)) 
	plt.imshow(neighbors[i], cmap='gray', interpolation='nearest')
	plt.title(dists[i]) 
	plt.axis('off') 

	plt.subplot2grid((2,7), (1,i+2)) 
	plt.imshow(neighbors1[i], cmap='gray', interpolation='nearest')
	plt.title(dists1[i]) 
	plt.axis('off') 

plt.tight_layout() 

plt.savefig(distMode+"_neighbors_"+TorS+".png")

#plt.show() 

