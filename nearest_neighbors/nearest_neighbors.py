#import matplotlib as mpl 
#mpl.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix
import itertools 
import numpy as np
import cv2 
import ot

# From Shaoib 
from high_dim_tests import MaximumMeanDis_mix as MMD
from sinkhorn_div import divergence as SinkDiv 


## Configurations 

inDir = "/home/zimani/GenNets/npy_files/" 

# "track" or "shower"
TorS = "track" 

genFile = "gen_epoch50" 
gens = np.load(inDir+genFile+"_"+TorS+"s.npy") 

larFile = "larcv_png_64_train" 
lars = np.load(inDir+larFile+"_"+TorS+"s.npy") 

nNeighbors = 5 

# "near" or "far" -est neighbors 
nnDist = "near" 

# l2 or EMD (or emperical distributions: MMD and SinkDiv)
distMode = "EMD" 

# Choose which generated image to use -- select by eye 
imgNum = 178
img = gens[imgNum,:] 
if 0: 
	fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
	for i, ax in enumerate(axes.flatten()): 
		ax.imshow(gens[imgNum+i,:], cmap='gray', interpolation='none')
		ax.set_title(str(imgNum+i))
		ax.axis('off') 
	plt.tight_layout() 
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

if distMode == 'EMD':
	# OpenCV EMD setup: signature matrix 
	xx, yy = np.meshgrid(np.arange(64), np.arange(64))
	xx = xx.ravel()
	yy = yy.ravel()
	img = img / np.sum(img) 
	imgSig = np.vstack((img.ravel(), xx, yy)).T.astype(np.float32)

	# POT EMD setup: L2 Distance matrix 
	p = np.array(list(itertools.product(np.arange(64), repeat=2))) 
	M = np.array(distance_matrix(p, p, p=2))

if nnDist == 'near': 
	nn = -1 
else: 
	nn = 1 

# Initializations
larDists = [] 
genDists = [] 

# Compare to training data 
for lar in lars: 
	if distMode == 'l2': 
		larDist = int(np.linalg.norm(img - lar))
	elif distMode == 'MMD': 
		larDist = round(MMD(img, lar, sigma_list).item(),6) 
	elif distMode == "SinkDiv": 
		larDist = round(SinkDiv(img, lar, eps).item(), 2)
	elif distMode == 'EMD': 
		# OpenCV
		lar = lar / np.sum(lar) 
		larSig = np.vstack((lar.ravel(), xx, yy)).T.astype(np.float32)
		larDist  = cv2.EMD(imgSig, larSig, cv2.DIST_L2)[0]
		# POT - not work? 
		#larDist = ot.emd2(img.ravel()/np.sum(img), lar.ravel()/np.sum(lar), M) 
	else: 
		print("Error: invalid distance mode") 
		exit() 

	larDists.append(larDist) 

larDists = np.asarray(larDists)
if distMode == 'EMD': 
	np.save("./emd_npy/emd_dists_lartpc_"+TorS+"_"+str(imgNum)+".npy", larDists) 
if distMode == 'l2':  
	np.save("./l2_npy/l2_dists_lartpc_"+TorS+"_"+str(imgNum)+".npy", genDists) 

# Sort nearest neighbors 
idxSortL = (nn*larDists).argsort() 
larDists = larDists[idxSortL[::-1]] 
larSort = lars[idxSortL[::-1]] 

# Compare to generated data
for gen in gens: 
	if distMode == 'l2': 
		genDist = int(np.linalg.norm(img - gen))
	if distMode == 'MMD': 
		genDist = round(MMD(img, gen, sigma_list).item(), 6)
	if distMode == "SinkDiv": 
		genDist = round(SinkDiv(img, gen, eps).item(), 2)
	if distMode == 'EMD': 
		# OpenCv
		gen = gen / np.sum(gen) 
		genSig = np.vstack((gen.ravel(), xx, yy)).T.astype(np.float32)
		genDist = cv2.EMD(imgSig, genSig, cv2.DIST_L2)[0]
		# POT - not work? 
		#genDist = ot.emd2(img.ravel()/np.sum(img), gen.ravel()/np.sum(gen), M) 

	genDists.append(genDist) 

genDists = np.asarray(genDists) 
if distMode == 'EMD': 
	np.save("./emd_npy/emd_dists_gen_"+TorS+"_"+str(imgNum)+".npy", genDists) 
if distMode == 'l2':  
	np.save("./l2_npy/l2_dists_gen_"+TorS+"_"+str(imgNum)+".npy", genDists) 

# Sort nearest neighbors 
idxSortG = (nn*genDists).argsort() 
genDists = genDists[idxSortG[::-1]] 
genSort = gens[idxSortG[::-1]] 

# Remove self match - need adjust nNeighbors
if nnDist == 'near': 
	genDists = genDists[1:] 
	genSort = genSort[1:] 

#print("Done") 
#exit() 

## Plotting 

plt.figure(figsize=(14,4)) 

plt.subplot2grid(shape=(2, 7), loc=(0,0), colspan=2, rowspan=2) 
plt.imshow(img, cmap='gray', interpolation='nearest') 
plt.title("Generated") 
plt.axis('off') 

for i in range(nNeighbors): 
	plt.subplot2grid((2,7), (0,i+2)) 
	plt.imshow(larSort[i], cmap='gray', interpolation='nearest')
	plt.title(larDists[i]) 
	plt.axis('off') 

	plt.subplot2grid((2,7), (1,i+2)) 
	plt.imshow(genSort[i], cmap='gray', interpolation='nearest')
	plt.title(genDists[i]) 
	plt.axis('off') 

plt.tight_layout() 

plt.savefig("./emd_plots/"+distMode+"_neighbors_"+TorS+"_"+str(imgNum)+".png")
#plt.savefig("./l2_plots/"+distMode+"_neighbors_"+TorS+"_"+str(imgNum)+".png")
#plt.savefig("./"+distMode+"_neighbors_"+TorS+"_"+str(imgNum)+".png")
print("Saved Fig") 

plt.show() 

