import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

MMD_mixed_train = np.load("./npy_files/MMD_mixed_train.npy") 
#MMD_tracks_train = np.load("./npy_files/MMD_tracks_train.npy") 
#MMD_showers_train = np.load("./npy_files/MMD_showers_train.npy") 

MMD_mixed_test = np.load("./npy_files/MMD_mixed_test.npy") 
#MMD_tracks_test = np.load("./npy_files/MMD_tracks_test.npy") 
#MMD_showers_test = np.load("./npy_files/MMD_showers_test.npy") 

Sink_mixed_train = np.load("./npy_files/Sink_mixed_train_1.npy") 
Sink_mixed_test = np.load("./npy_files/Sink_mixed_test_1.npy") 

W1_mixed_train = np.load("./npy_files/W1_mixed_train.npy") 
W1_mixed_test = np.load("./npy_files/W1_mixed_test.npy") 

## GoF.append(np.array([score, epoch, gen.shape[0]])) 	

## Sigma list = 2^10 : 2^15

# Skip first few epochs 
strt = 1

epochs = MMD_mixed_train[strt:,1]

# Plotting 
#plt.figure(figsize=(10,6)) 
plt.rcParams.update({'font.size': 18}) 
fig, ax1 = plt.subplots(1, 1, figsize=(10,6)) 
ax2 = ax1.twinx()

l1, = ax1.plot(epochs, MMD_mixed_train[strt:,0], '-bo', label="MMD Training Set")# ("+str(int(MMD_mixed_train[0,2]))+")") 
l2, = ax1.plot(epochs, MMD_mixed_test[strt:,0], ':bo', label="MMD Validation Set")# ("+str(int(MMD_mixed_test[0,2]))+")") 

l3, = ax2.plot(epochs, Sink_mixed_train[strt:,0], '-ro', label="SinkDiv Training Set")# ("+str(int(Sink_mixed_train[0,2]))+")") 
l4, = ax2.plot(epochs, Sink_mixed_test[strt:,0], ':ro', label="SinkDiv Validation Set")# ("+str(int(Sink_mixed_test[0,2]))+")") 

l5, = ax2.plot(epochs, W1_mixed_train[strt-1:,0], '-go', label="Wasserstein-1 Training Set")# ("+str(int(W1_mixed_train[0,2]))+")") 
l6, = ax2.plot(epochs, W1_mixed_test[strt-1:,0], ':go', label="Wasserstein-1 Validation Set")# ("+str(int(W1_mixed_test[0,2]))+")") 

fig.suptitle("High Dimensional Goodness of Fit Tests", y=0.92) 
ax1.set_xlabel("Epochs") 
ax1.set_ylabel("MMD") 
#ax1.set_ylabel("Sinkhorn Divergence Metric") 
ax2.set_ylabel("Wasserstein-1 and SinkDiv") 
ax1.set_yscale("log")
ax2.set_yscale("log")

ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())


#plt.xticks(epochs)
#plt.xticks([0,10,20,30,40,50,60,100,150,200,250,300]) 
plt.xticks([10,30,50,100,150,200,250,300]) 


print("MMD[2:] Range:", min(MMD_mixed_train[2:,0]), max(MMD_mixed_test[2:,0]))
print("Epochs:", epochs) 

#ax1.set_ylim(0.025, 0.55)
#ax2.set_ylim(1100, 1205)

ax1.set_ylim(0.00008, 0.1) 
ax2.set_ylim(1075, 1400)

ax1.legend(handles=[l1, l2, l3, l4, l5, l6])#, loc=(0.68,0.55))
plt.tight_layout()

plt.savefig("./hists/GoF.png") 

plt.show() 

