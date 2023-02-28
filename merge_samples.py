import numpy as np
from PIL import Image
import os 

inDir = "/home/zimani/score_sde_pytorch/larcv_png64_workdir/eval/ckpt_10/"

outFile = "/home/zimani/score_sde_pytorch/gen_epoch100_orig.npy"

#newData = np.load(inDir+"samples_0.npz")['samples'] 

newData = []

for i in range(1,1000):
	filename = inDir+"samples_"+str(i)+".npz" 
	if os.path.exists(filename): 
		sample = np.load(filename) 
		newData.append(sample['samples'])
		#newData = np.concatenate((newData, sample['samples']), axis=0)
	else: 
		break


#newData.reshape(newData.shape[0]*newData.shape[1], 64, 64, 3) 

print(np.array(newData).shape)
np.save(outFile, newData) 


