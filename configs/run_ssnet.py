import os,sys
import numpy as np
import torch

# Load other files/modules - need run from parent folder 
sys.path.append("uresnet_pytorch/") 
from uresnet.flags import URESNET_FLAGS
from uresnet.models.uresnet_dense import UResNet

import ROOT as rt
from ROOT import std
rt.gSystem.Load("libssnetdata")
from ROOT import ssnet

# from torchinfo import summary # Visulize Model
from tqdm import tqdm # progress bar 
from absl import flags as fg 
from absl import app 

"""
Run SSNet on images to
	1) Save ROOT histogram 
	2) Separate track and shower events
	3) Save activations for FID (optional)
Run from parent folder as "python py/run_ssnet.py" 
"""

set_outName = "gen_epoch50"
set_inFile = "/home/zimani/key_generated/gen_epoch50.npy" 
set_outPath = "/home/zimani/"

FLAGS = fg.FLAGS
fg.DEFINE_bool('saveROOT', True, 'Save ROOT histogram')
fg.DEFINE_bool('saveEvents', True, 'Save track and shower events seperatly')
fg.DEFINE_bool('saveFID', False, 'Save FID activations')
fg.DEFINE_string('outPath', set_outPath, 'Path to desired output file location for hists and track/shower images')
fg.DEFINE_string('outPathFID', None, 'Path to desired output file location')
fg.DEFINE_string('outName', set_outName, 'Name of output files')
fg.DEFINE_string('inFile', set_inFile, 'File to process from external source')

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

# Model
model = UResNet(flags).to(DEVICE)
device_map = {"cuda:0":flags.DEVNAME,
			  "cuda:1":flags.DEVNAME}
checkpoint = torch.load( flags.checkpoint_file, map_location=device_map )
model.load_state_dict( checkpoint["state"] )
model.eval()

track_array = [] 
shower_array = []

# Setup Activations for FID
activation = {} 
def get_activation(name): 
	def hook(model, input, output):
		activation[name] = output.detach() 
	return hook 
layerName = 'myAct' 
model.double_resnet[2].register_forward_hook(get_activation(layerName)) 
actArray = [] 

def main(argv): 

	# Sanatize inputs (minimally)
	outPath = FLAGS.outPath 
	if outPath[-1] != "/": 
		outPath += "/" 
	outPathFID = FLAGS.outPathFID
	if FLAGS.saveFID and FLAGS.outPathFID is None: 
		outPathFID = FLAGS.outPath 
		if outPathFID[-1] != "/": 
			outPathFID += "/" 
	outName = FLAGS.outName
	if outName[-4:-1] == ".np":
		outName = outName[:-4] 
	
	# Load Samples
	images = np.load(FLAGS.inFile)
	print("Loaded", FLAGS.inFile, images.shape) 
	
	# Switch to pytorch form
	images = images.reshape( (images.shape[0],1,64,64) ).astype(np.float32)
	THRESHOLD = 0.2

	# Setup Output
	if FLAGS.saveROOT: 
		rootName = "ssnet_hists_%s.root"%(FLAGS.outName)
		fout = rt.TFile(outPath+rootName,"recreate")

	filler = ssnet.FillScoreHist() # c++ routine to speed up filling histogram
	filler.define_hists()
	PIXCLASSES = ["bg","shower","track"]

	# Define Loop Parameters 
	NIMAGES = images.shape[0]
	NITERS = int(NIMAGES/flags.BATCH_SIZE)
	softmax = torch.nn.Softmax( dim=1 )
	
	for iiter in tqdm(range(NITERS)):
		
		start_index = flags.BATCH_SIZE*iiter
		end_index = flags.BATCH_SIZE*(iiter+1)

		if end_index>NIMAGES:
			end_index = NIMAGES

		bsize = end_index-start_index

		# Prep Input Tensor
		in_images = images[start_index:end_index,:,:,:]
		images_t = torch.from_numpy( in_images ).float().to(DEVICE)
		
		# Run SSNet on batch of (16) images 
		with torch.no_grad():
			out_t = model(images_t)

		# Get Activations for FID 
		if FLAGS.saveFID: 
			activations = activation[layerName]
			activeNPY = activations.detach().cpu().numpy() 
			
			# Append to Python array (works but slow)
			actArray.append(activeNPY)
		
		# Format prediction labels 
		pred_t = softmax(out_t)
		pred_t[ pred_t>0.999 ] = 0.999
		pred_t = pred_t.cpu().numpy()
  
		for ib in range(bsize):

			## Fill histograms via the filler clas 
			npix = filler.fillInternalHists( in_images[ib,0,:,:], pred_t[ib,:,:,:], THRESHOLD )

			# Track or shower classification and extraction
			# There is probably a more efficent way...

			# Seperate label probabilities 
			bkgrnd = pred_t[ib,:,:,:][0] 
			shower = pred_t[ib,:,:,:][1]
			track = pred_t[ib,:,:,:][2]
			
			# Select pixels that are most likely not background 
			shower_pix = np.greater(shower, bkgrnd) 
			track_pix = np.greater(track, bkgrnd) 

			# Only keep non-background pixels 
			mask = np.logical_or(shower_pix, track_pix) 
			shower = shower[mask]
			track = track[mask] 

			# Count number of track or shower pixels 
			track_gt_shower = np.greater(track, shower) 
			num_tracks = np.sum(track_gt_shower) # number track pixels 
			num_showers = np.sum(np.logical_not(track_gt_shower)) # number shower pixels 

			# Append current image to appropriate Python array (works but slow)
			if num_tracks > num_showers: 
				track_array.append(in_images[ib,0,:,:])
			else: 
				shower_array.append(in_images[ib,0,:,:]) 
			
	# Write histograms to disk
	if FLAGS.saveROOT: 
		fout.Write() 
		fout.Close()

	# Save Track and Shower Events Separately 
	if FLAGS.saveEvents: 
		np.save(outPath+outName+"_tracks.npy", track_array)
		np.save(outPath+outName+"_showers.npy", shower_array)
		print("Saved:", outPath+outName+"_tracks(_showers).npy" )

	# Save Layer Activations for FID Metric 
	if FLAGS.saveFID: 
		FID = np.asarray(actArray)
		FID = FID.reshape((FID.shape[0]*FID.shape[1],)+FID.shape[2:]) 
		np.save(outPathFID+outName+"_FID.npy", FID) 
		print("Saved FID with shape:", FID.shape) 

	return

if __name__ == '__main__': 
	app.run(main)
