import numpy as np
from PIL import Image
import os 

inDir = "/home/zimani/GenNets/energy_analysis/npy_files/" 
outDir = "./samples/"

#inFile = "gen_epoch100_tracks.npy" 
inFile = "larcv_png_64_train_tracks.npy" 

#outFile = "gen_epoch100_tracks_1k.npy"
outFile = "larcv_png_64_train_tracks_1k.npy" 

samples = np.load(inDir+inFile) 

print(samples.shape) 

trimmed = samples[0:1000,:,:] 

print(trimmed.shape)

np.save(outDir+outFile, trimmed)  
