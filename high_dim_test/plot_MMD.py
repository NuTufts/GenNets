import matplotlib.pyplot as plt
import numpy as np

MMDs_mixed_train = np.load("./npy_files/MMD_mixed_train.npy") 
MMDs_tracks_train = np.load("./npy_files/MMD_tracks_train.npy") 
MMDs_showers_train = np.load("./npy_files/MMD_showers_train.npy") 

MMDs_mixed_test = np.load("./npy_files/MMD_mixed_test.npy") 
MMDs_tracks_test = np.load("./npy_files/MMD_tracks_test.npy") 
MMDs_showers_test = np.load("./npy_files/MMD_showers_test.npy") 

## Hardcoded 
epochs = [20, 30, 40, 50, 100, 150, 300] 

## Include number of samples as n in legend 

# Plotting 
plt.figure(figsize=(10,6)) 
plt.plot(epochs, MMDs_mixed_train, '-ko', label="Mixed Train (15000)") 
plt.plot(epochs, MMDs_tracks_train, '-ro', label="Track Train (15000)") 
plt.plot(epochs, MMDs_showers_train, '-bo', label="Shower Train (15000)") 
plt.plot(epochs, MMDs_mixed_test, ':ko', label="Mixed Val (4000)") 
plt.plot(epochs, MMDs_tracks_test, ':ro', label="Track Val (3144)") 
plt.plot(epochs, MMDs_showers_test, ':bo', label="Shower Val (856)") 
plt.title("MMD Results") 
plt.xlabel("Epochs") 
plt.ylabel("MMD Score") 
#plt.ylim(0.75*min(MMDs_train), 1.5*max(MMDs_train)) 
plt.legend() 
plt.tight_layout()

plt.savefig("./hists/MMD_all.png") 

plt.show() 

