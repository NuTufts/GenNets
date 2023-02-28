from PIL import Image
import numpy as np
import os


path = "/home/zimani/particle_datasets/track_dataset/train_track/"
path = "/home/zimani/particle_datasets/larcv_png_64_grayscale/larcv_png_64_train/" 
tracks = [] 

for trackFile in os.listdir(path):
	png = Image.open(path+trackFile).convert('L')
	track = np.asarray(png)
	tracks.append(track) 

np.save("larcv_png_64_train.npy", tracks)
