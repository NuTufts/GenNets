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

W1_mixed_train = np.load("./npy_files/W1_mixed_train.npy") 
W1_mixed_test = np.load("./npy_files/W1_mixed_test.npy") 



#GoF.append(np.array([score, epoch, gen.shape[0]])) 	

## Sigma list = 2^10 : 2^15


epochs = MMD_mixed_train[:,1]

# Plotting 
plt.figure(figsize=(10,6)) 
#plt.plot(epochs, MMD_mixed_train[:,0], '-ko', label="MMD Train ("+str(int(MMD_mixed_train[0,2]))+")") 
plt.plot(epochs, Sink_mixed_train[:,0], '-ro', label="SinkDiv Train ("+str(int(Sink_mixed_train[0,2]))+")") 
plt.plot(epochs, W1_mixed_train[:,0], '-bo', label="Wasserstein-1 Train ("+str(int(W1_mixed_train[0,2]))+")") 
#plt.plot(epochs, MMD_mixed_test[:,0], ':ko', label="MMD Val ("+str(int(MMD_mixed_test[0,2]))+")") 
plt.plot(epochs, Sink_mixed_test[:,0], ':ro', label="SinkDiv Val ("+str(int(Sink_mixed_test[0,2]))+")") 
plt.plot(epochs, W1_mixed_test[:,0], ':bo', label="Wasserstein-1 Val ("+str(int(W1_mixed_test[0,2]))+")") 
plt.title("High Dimensional GoF Tests") 
plt.xlabel("Epochs") 
plt.ylabel("Score") 
plt.yscale("log")
plt.legend() 
plt.tight_layout()

plt.savefig("./hists/GoF_most.png") 

plt.show() 

