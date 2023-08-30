import os,sys

"""
run ssnet on test images and save output to a histogram.
"""
import numpy as np
import torch

sys.path.append("./uresnet_pytorch/") 
from uresnet.flags import URESNET_FLAGS
from uresnet.models.uresnet_dense import UResNet

import ROOT as rt
from ROOT import std

x = rt.gSystem.Load("libssnetdata")

from ROOT import ssnet

# Visulize Model
from torchinfo import summary

from absl import flags as fg 
from absl import app 

### Start Configs 


set_epoch = 50 
set_outFile = "gen_epoch50"
set_inFile = "" 

FLAGS = fg.FLAGS

#fg.DEFINE_bool('genSamples', True, 'Process epoch from Score Network')
#fg.DEFINE_integer('epoch', set_epoch, 'Training epoch to process')
#fg.DEFINE_string('genPath', "/home/zimani/score_sde_pytorch/larcv_png64_workdir", 'Score Network working directory')
#fg.DEFINE_string('outFile', set_outFile, 'Name of output numpy file')
#fg.DEFINE_integer('numSamples', 50000, 'Number of generated images to process')
#fg.DEFINE_string('inFile', set_inFile, 'File to process from external source')


#data_to_run = {"larcv_png_64_test": "/home/zimani/key_generated/larcv_png_64_test.npy"}
#data_to_run = {"larcv_png_64_train": "/home/zimani/key_generated/larcv_png_64_train.npy"}
#data_to_run = {"VQVAE":"/home/zimani/key_generated/VQVAE.npy"}
#data_to_run = {"paul_results":"/home/zimani/key_generated/paul_results.npy"}

zpoch = "50"
data_to_run = {"gen_epoch"+zpoch:"/home/zimani/key_generated/gen_epoch"+zpoch+".npy"}
#data_to_run = {"gen_epoch"+zpoch:"/home/zimani/GenNets/gen_epoch"+zpoch+".npy"}

#outPath = "/home/zimani/GenNets/npy_files/"
outPath = "/home/zimani/"

# Save ROOT Histograms 
saveROOT = True

# Save Track and Shower events
saveEvents = True 

# Save Activations for FID Metric
saveFID = False

# Get name for event 
outFileName = list(data_to_run.keys())
if len(outFileName) > 1: 
	print("Name Error") 
	exit() 
outFileName = str(outFileName[0]) 

### End Configs  


track_array = [] 
shower_array = []

# Configure network and load it
flags = URESNET_FLAGS()
flags.DATA_DIM = 2
flags.URESNET_FILTERS = 16
flags.URESNET_NUM_STRIDES = 4
flags.SPATIAL_SIZE = 256
flags.NUM_CLASS = 3 # bg, shower, track
flags.LEARNING_RATE = 1.0e-3
flags.WEIGHT_DECAY = 1.0e-4
flags.BATCH_SIZE = 16
flags.checkpoint_file = "ssnet.dlpdataset.forSimDL2021.tar" # 16 features, images conditioned
flags.DEVNAME = "cuda:0"

DEVICE = torch.device(flags.DEVNAME)

# MODEL
model = UResNet(flags).to(DEVICE)
device_map = {"cuda:0":flags.DEVNAME,
			  "cuda:1":flags.DEVNAME}
checkpoint = torch.load( flags.checkpoint_file, map_location=device_map )
model.load_state_dict( checkpoint["state"] )
model.eval()

## FID 
activation = {} 
def get_activation(name): 
	def hook(model, input, output):
		activation[name] = output.detach() 
	return hook 
layerName = 'myAct' 

#layerObj = model.double_resnet[3].children() 
#print(list(layerObj)) 

# FID1 (Use this) 
model.double_resnet[2].register_forward_hook(get_activation(layerName)) 

# FID2 (Not this) 
#model.decode_double_resnet[0].register_forward_hook(get_activation(layerName)) 


## Print Model Summary 
#inputShape = (16, 1, 64, 64) 
#summary(model, inputShape) 
#exit() 

actArray = [] 


## End FID 


def run_sample( input_file, output_file ):

	# LOAD SAMPLES
	images  = np.load(input_file)
	print(images.shape)
	
	# switch to pytorch form
	images = images.reshape( (images.shape[0],1,64,64) ).astype(np.float32)
	THRESHOLD = 0.2
	#images *= 255.0/10.0
	print(images.shape,images.dtype)

	# SETUP OUTPUT
	if saveROOT: 
		fout = rt.TFile(output_file,"recreate")

	filler = ssnet.FillScoreHist() # c++ routine to speed up filling histogram
	filler.define_hists()
	PIXCLASSES = ["bg","shower","track"]

	# DEFINE LOOP PARAMS
	NIMAGES = images.shape[0]
	NITERS = int(NIMAGES/flags.BATCH_SIZE)
	softmax = torch.nn.Softmax( dim=1 )

	# Intitalize Numpy array 
	
	for iiter in range(NITERS):

		print("[ITER ",iiter," of ",NITERS,"]")
		
		start_index = flags.BATCH_SIZE*iiter
		end_index = flags.BATCH_SIZE*(iiter+1)

		if end_index>NIMAGES:
			end_index = NIMAGES

		bsize = end_index-start_index

		# prep input tensor
		in_images = images[start_index:end_index,:,:,:]
		images_t = torch.from_numpy( in_images ).float().to(DEVICE)
		
		# Run SSNet on batch of (16) images 
		with torch.no_grad():
			out_t = model(images_t)

		## FID 
		if saveFID: 
			activations = activation[layerName]
			activeNPY = activations.detach().cpu().numpy() 

			#print("Activations:", activeNPY.shape) 
			#testA = activeNPY.reshape(activeNPY.shape[0], -1) 
			#print(testA.shape) 
			#print(np.min(testA), np.max(testA)) 
			#exit() 

			actArray.append(activeNPY)
		
		# Format prediction labels 
		pred_t = softmax(out_t)
		pred_t[ pred_t>0.999 ] = 0.999
		#print("pred_t shape: ",pred_t.shape)
		pred_t = pred_t.cpu().numpy()

		# fill histograms via the fillter class
		nabove_tot = 0
		for ib in range(bsize):
			npix = filler.fillInternalHists( in_images[ib,0,:,:], pred_t[ib,:,:,:], THRESHOLD )

			## Track or Shower Image Classification / Extraction   
			##   Zev code: works but inefficent 
			this_image = in_images[ib,0,:,:] 

			bkgrnd = pred_t[ib,:,:,:][0] 
			shower = pred_t[ib,:,:,:][1]
			track = pred_t[ib,:,:,:][2]
			
			shower_pix = np.greater(shower, bkgrnd) 
			track_pix = np.greater(track, bkgrnd) 

			mask = np.logical_or(shower_pix, track_pix) 

			shower = shower[mask]
			track = track[mask] 

			track_gt_shower = np.greater(track, shower) #number of track pixels 
			num_tracks = np.sum(track_gt_shower) ## number track pixels 
			num_showers = np.sum(np.logical_not(track_gt_shower)) ## number shower pixels 

			if num_tracks > num_showers: 
				track_or_shower = "track"  
				track_array.append(this_image)
			else: 
				track_or_shower = "shower"  
				shower_array.append(this_image) 
			## End Zev code 
			print(pred_t[ib,:,:,:].shape) 
			exit() 

			nabove_tot += npix
			#print("Num above thresh in batch: ",nabove_tot," per image: ",nabove_tot/float(bsize))

	# Write histograms to disk
	if saveROOT: 
		fout.Write() 
		fout.Close()

	# Save Track and Shower Events Separatly 
	if saveEvents: 
		np.save(outPath+outFileName+"_tracks.npy", track_array) 
		np.save(outPath+outFileName+"_showers.npy", shower_array) 

		#np.save(outPath+"gen_epoch"+zpoch+"_tracks", track_array) 
		#np.save(outPath+"gen_epoch"+zpoch+"_showers", shower_array) 

		#np.save(outPath+"larcv_png_64_train_tracks", track_array) 
		#np.save(outPath+"larcv_png_64_train_showers", shower_array) 

		#np.save(outPath+"VQVAE_tracks", track_array) 
		#np.save(outPath+"VQVAE_showers", shower_array) 

		#np.save(outPath+"paul_results_tracks", track_array) 
		#np.save(outPath+"paul_results_showers", shower_array) 
		pass 

	# Save Layer Activations for FID Metric 
	if saveFID: 
		FID = np.asarray(actArray)
		FID = FID.reshape((FID.shape[0]*FID.shape[1],)+FID.shape[2:]) 
		print("FID Shape:", FID.shape) 
		np.save("/home/zimani/GenNets/FID/"+outFileName+"_FID.npy", FID) 

	return


for sample_name, data_file in data_to_run.items():

	print("=====================================")
	print(" RUN ",sample_name,": ",data_file)
	print("=====================================")
	outfile = "ssnet_hists_%s.root"%(sample_name)
	run_sample( data_file, outPath+outfile )


	
