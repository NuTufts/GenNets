import matplotlib.pyplot as plt 
import numpy as np


tracks = np.load("./energy_analysis/npy_files/gen_epoch1_tracks.npy")
showers = np.load("./energy_analysis/npy_files/gen_epoch1_showers.npy")

plt.imshow(tracks[0], cmap='gray') 
plt.show()
