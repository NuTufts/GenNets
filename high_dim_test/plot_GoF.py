import matplotlib.pyplot as plt
import numpy as np

MMD_mixed_train = np.load("./npy_files/MMD_mixed_train.npy") 
#MMD_tracks_train = np.load("./npy_files/MMD_tracks_train.npy") 
#MMD_showers_train = np.load("./npy_files/MMD_showers_train.npy") 

MMD_mixed_test = np.load("./npy_files/MMD_mixed_test.npy") 
#MMD_tracks_test = np.load("./npy_files/MMD_tracks_test.npy") 
#MMD_showers_test = np.load("./npy_files/MMD_showers_test.npy") 

Sink_mixed_train = np.load("./npy_files/Sink_mixed_train.npy") 
Sink_mixed_test = np.load("./npy_files/Sink_mixed_test.npy") 

#W1_mixed_train = np.load("./npy_files/W1_mixed_train.npy") 
#W1_mixed_test = np.load("./npy_files/W1_mixed_test.npy") 

## GoF.append(np.array([score, epoch, gen.shape[0]])) 	

## Sigma list = 2^10 : 2^15

# Skip first few epochs 
strt = 0

epochs = MMD_mixed_train[strt:,1]

# Plotting 
#plt.figure(figsize=(10,6)) 
fig, ax1 = plt.subplots(1, 1, figsize=(10,6)) 
ax2 = ax1.twinx()

l1, = ax1.plot(epochs, MMD_mixed_train[strt:,0], '-bo', label="MMD Train ("+str(int(MMD_mixed_train[0,2]))+")") 
l2, = ax2.plot(epochs, Sink_mixed_train[strt:,0], '-ro', label="SinkDiv Train ("+str(int(Sink_mixed_train[0,2]))+")") 
#l3, = ax1.plot(epochs, W1_mixed_train[strt:,0], '-bo', label="Wasserstein-1 Train ("+str(int(W1_mixed_train[0,2]))+")") 
l4, = ax1.plot(epochs, MMD_mixed_test[strt:,0], ':bo', label="MMD Val ("+str(int(MMD_mixed_test[0,2]))+")") 
l5, = ax2.plot(epochs, Sink_mixed_test[strt:,0], ':ro', label="SinkDiv Val ("+str(int(Sink_mixed_test[0,2]))+")") 
#l6, = ax1.plot(epochs, W1_mixed_test[strt:,0], ':bo', label="Wasserstein-1 Val ("+str(int(W1_mixed_test[0,2]))+")") 

fig.suptitle("High Dimensional Goodness of Fit Tests") 
ax1.set_xlabel("Epochs") 
ax1.set_ylabel("MMD Metric") 
ax2.set_ylabel("Sinkhorn Divergence Metric") 
ax1.set_yscale("log")
ax2.set_yscale("log")

print(min(MMD_mixed_train[2:,0]), max(MMD_mixed_test[2:,0]))

ax1.set_ylim(0.025, 0.55)
ax2.set_ylim(1100, 1205)

ax1.legend(handles=[l1, l2, l4, l5], loc=(0.68,0.55))
plt.tight_layout()

plt.savefig("./hists/GoF_all.png") 

#plt.show() 

