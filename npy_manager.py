import numpy as np
from PIL import Image
import os


## Parameters 
genSamples = False 

ckpt = 6

outfile = "gen_epoch50.npy"  
## End Parameters 


# Get samples from generated checkpoint (formerly merge_samples.py)  
if genSamples: 

	genDir = "/home/zimani/score_sde_pytorch/larcv_png64_workdir/eval/ckpt_"+str(ckpt)+"/"

	newData = []
	
	# Merge all batches into single array 
	for i in range(1,1000):
		filename = genDir+"samples_"+str(i)+".npz" 
		if os.path.exists(filename): 
			sample = np.load(filename) 
			newData.append(sample['samples'])
		else: 
			break

	samples = np.asarray(newData) 

# Load existing numpy file 
else:	

	inDir = "/home/zimani/"

	#inFile = "ckpt_3_samples.npy"
	#inFile = "larcv_png_64_train_grayscale.npy"
	inFile = "gen_epoch300_orig.npy"
	
	# Remove last underscore from name 
	outFile = "_".join(inFile.split("_")[:-1])+".npy"

	samples = np.load(inDir+inFile)


# Optional: trim number of samples to 32 
#trimmed = samples[0:32,:,:] 

# Mask out noise 
samples[np.where(samples <= 4)] = 0

# Number of images desired
maxSamples = 50000
totSamples = 0 

# Batch size 
bSize = samples.shape[1]  
batches = int(maxSamples/bSize) + 1 #rounding 

# Convert to grayscale npy files
saveGray = True
grayImages = []

# Option: save a sample png 
saveSamples = False
sampleDir = ""

# Iterate batches 
for j in range(0, batches): 

	# Iterate image in batch 
	for i in range(0, bSize):

		try: 
			image = samples[j][i]
			totSamples += 1
		except: 
			break

		# Open image using PIL
		im = Image.fromarray(image)

		# Convert to grayscale (lossy??)
		im = im.convert('L')

		# Sample Image 
		if saveSamples and i==10 and j==10:
			im.save(sampleDir+"image_"+str(i)+".png")

		# Insert single channel (64, 64) -> (64, 64, 1)
		im = np.expand_dims(im, axis=-1)

		# Create new array of grayscale images
		grayImages.append(im)
		
		# Limit number of samples 
		if totSamples == maxSamples: 
			pass 
			#break 

# Save grayscaled npy 
if saveGray: 
	np.save(outFile, grayImages) 

print(totSamples, "Images Processed")     

