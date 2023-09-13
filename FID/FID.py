from scipy import linalg
from absl import flags 
from absl import app
import numpy as np

set_inDir = "./ssnet_activations/"
set_trainFile = "larcv_png_64_train_FID.npy"
set_valFile = "larcv_png_64_test_FID.npy"

FLAGS = flags.FLAGS 
flags.DEFINE_string('inDir', set_inDir, "Directory containing SSNet activations in numpy format")
flags.DEFINE_string('trainFile', set_trainFile, "Numpy file containing training dataset activations")
flags.DEFINE_string('valFile', set_valFile, "Numpy file containing validation dataset activations") 

def main(argv): 

	# Names of activations to compare 
	# 	Note: _FID.npy is appended when loading 
	fileNames = ["gen_epoch10", 
		"gen_epoch20", 
		"gen_epoch30", 
		"gen_epoch40", 
		"gen_epoch50", 
		"gen_epoch60", 
		"gen_epoch100", 
		"gen_epoch150", 
		"gen_epoch300", 
		"VQVAE"] 

	larcvT = np.load(FLAGS.inDir+FLAGS.trainFile) 
	larcvT = larcvT.reshape(larcvT.shape[0], -1) 
	larcvV = np.load(FLAGS.inDir+FLAGS.valFile) 
	larcvV = larcvV.reshape(larcvV.shape[0], -1) 

	# Downsample training larcv dataset to larcv dataset size   
	if larcvT.shape[0] > larcvV.shape[0]: 
		idxLar = np.random.choice(np.arange(larcvT.shape[0]), size=larcvV.shape[0], replace=False) 
		larcvT = larcvT[idxLar] 

	# Calculate mean and covaraince matrix for train and val
	sigmaT = np.cov(larcvT, rowvar=False) 
	muT = larcvT.mean(axis=0) 
	sigmaV = np.cov(larcvV, rowvar=False) 
	muV = larcvV.mean(axis=0) 

	FIDs = [] 

	print("FileName, Training FID, Validation FID") 

	for i, fileName in enumerate(fileNames):

		gen = np.load(FLAGS.inDir+fileName+"_FID.npy") 
		gen = gen.reshape(gen.shape[0], -1) 
		
		# Downsample generated images to larcv comparison size 
		if larcvV.shape[0] != gen.shape[0]: 
			idx = np.random.choice(np.arange(gen.shape[0]), size=larcvV.shape[0], replace=False)  
			gen = gen[idx] 

		sigma = np.cov(gen, rowvar=False) 
		mu = gen.mean(axis=0) 
		
		# Training FID 
		# calculate sum squared difference between means
		ssdiffT = np.sum((muT - mu)**2.0)
		# calculate sqrt of product between cov
		covmeanT = linalg.sqrtm(sigmaT.dot(sigma))
		# check and correct imaginary numbers from sqrt
		if np.iscomplexobj(covmeanT):
			covmeanT = covmeanT.real
		# calculate score
		fidT = ssdiffT + np.trace(sigmaT + sigma - 2.0 * covmeanT)

		# Repeat for validation FID
		ssdiffV = np.sum((muV - mu)**2.0)
		covmeanV = linalg.sqrtm(sigmaV.dot(sigma))
		if np.iscomplexobj(covmeanV):
			covmeanV = covmeanV.real
		fidV = ssdiffV + np.trace(sigmaV + sigma - 2.0 * covmeanV)

		print(fileName, fidT, fidV) 
		FIDs.append(fileName, fidT, fidV) 

	np.save("FID_values.npy", np.array(FIDs))  

if __name__ == '__main__': 
	app.run(main) 



