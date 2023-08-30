import numpy as np
from PIL import Image
from absl import flags 
from absl import app
import os 

set_epoch = 50
set_outFile = "gen_epoch50.npy"  
set_inFile = "/home/zimani/VQVAE_orig.npy"

FLAGS = flags.FLAGS 
flags.DEFINE_bool('genSamples', True, 'Process epoch from Score Network') 
flags.DEFINE_integer('epoch', set_epoch, 'Training epoch to process') 
flags.DEFINE_string('genPath', "/home/zimani/score_sde_pytorch/larcv_png64_workdir", 'Score Network working directory') 
flags.DEFINE_string('outFile', set_outFile, 'Name of output numpy file')  
flags.DEFINE_integer('numSamples', 50000, 'Number of generated images to process') 
flags.DEFINE_string('inFile', set_inFile, 'File to process from external source, only if genSamples set to False') 

def main(argv): 

	# Dictionary: Epoch -> [workDir, ckpt]
	epochDict = {
		10 : ["_v1", 1],
		20 : ["_v1", 2],
		30 : ["_v1", 3],
		40 : ["_v1", 4],
		50 : ["_v1", 5],
		60 : ["_v1", 6],
		100 : ["_v1", 10],
		150 : ["_v1", 15],
		300 : ["_v2", 6]
	} 

	workDir = epochDict[FLAGS.epoch][0]
	ckpt = epochDict[FLAGS.epoch][1]

	# Get samples from generated checkpoint  
	if FLAGS.genSamples: 

		genDir = FLAGS.genPath+workDir+"/eval/ckpt_"+str(ckpt)+"/" 
		inFile = "" 

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

		# Confirm naming convention (optional) 
		if str(FLAGS.epoch) not in FLAGS.outFile: 
			print("Error: epoch not in outFile name") 
			exit() 
	
	# Load existing numpy file 
	else:	

		samples = np.load(FLAGS.inFile)
		
		# Old method: remove trailing _word from name 
		#inFile = "VQVAE_orig.npy"
		#outFile = "_".join(inFile.split("_")[:-1])+".npy"
		#samples = np.load("/home/zimani/"+inFile)

	# Mask out noise 
	if not "VQVAE" in inFile: 
		samples[np.where(samples <= 4)] = 0

	# Number of images desired
	maxSamples = FLAGS.numSamples
	totSamples = 0 

	# Batch size 
	bSize = samples.shape[1]  
	batches = int(maxSamples/bSize) + 1 #rounding 

	# Processed image array 
	procImages = []

	# Iterate batches 
	for j in range(0, batches): 

		# Iterate image in batch 
		for i in range(0, bSize):
			try: 
				image = samples[j][i]
				totSamples += 1
			except: 
				break
			
			# Remove single channel for processing 
			if image.shape == (64, 64, 1):
				image = np.squeeze(image)

			# Normalize and mask VQVAE
			if "VQVAE" in inFile: 
				if np.max(image) != 0: 
					image = image / np.max(image) 
				image *= 255 
				image[np.where(image <= 4)] = 0 

			# Open image using PIL
			im = Image.fromarray(image)

			# Convert to grayscale
			im = im.convert('L')

			# Insert single channel (64, 64) -> (64, 64, 1)
			im = np.expand_dims(im, axis=-1)

			# Add to array of processed images
			procImages.append(im)
			
			# Limit number of samples 
			if totSamples == maxSamples: 
				break 

	# Save processed events as npy 
	np.save(FLAGS.outFile, procImages) 

	print(totSamples, "images saved to", FLAGS.outFile)     

if __name__ == '__main__': 
	app.run(main) 
