import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np

inDir = "/home/zimani/GenNets/npy_files/"

larcvTracks = np.load(inDir+"larcv_png_64_train_tracks.npy")
larcvShowers = np.load(inDir+"larcv_png_64_train_showers.npy")
larcvs = np.vstack((larcvTracks, larcvShowers)) 
#larcvs = np.vstack((larcvTracks[0:larcvShowers.shape[0]], larcvShowers)) 

genTracks = np.load(inDir+"gen_epoch50_tracks.npy") 
genShowers = np.load(inDir+"gen_epoch50_showers.npy") 
gens = np.vstack((genTracks, genShowers)) 
#gens = np.vstack((genTracks[0:genShowers.shape[0]], genShowers)) 

# train or gen 
figName = "gen"

if figName == 'train': 
	events = larcvs 
	pltTitle = "Training Images" 
if figName == 'gen': 
	events = gens 
	pltTitle = "Generated Images (Epoch 50)" 

rows = 10
cols = 7

idx = np.random.choice(np.arange(events.shape[0]), size=rows*cols, replace=False)
samples = events[idx]  

plt.figure(figsize=(8, 12)) 
for i in range(rows*cols): 
	plt.subplot(rows, cols, i+1) 
	plt.imshow(samples[i], cmap='gray', interpolation='none') 
	plt.axis('off') 

plt.suptitle(pltTitle, fontsize=12) 

plt.tight_layout() 

#plt.savefig(figName+"_samples.png") 
plt.savefig(figName+"_full_page.png") 

plt.show() 

