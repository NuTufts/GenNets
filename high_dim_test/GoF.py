import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 
from absl import flags 
from absl import app
import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## File Imports 
from high_dim_tests import RankEnergy
from high_dim_tests import SoftRankEnergy
from high_dim_tests import two_sample_sinkdiv
from high_dim_tests import MaximumMeanDis_mix
from high_dim_tests import Wasserstein_1
from high_dim_tests import TwoSampleWTest

set_inDir = "/home/zimani/GenNets/npy_files/"
set_outDir = "./npy_files/"
set_GoF_test = "MMD"

FLAGS = flags.FLAGS 
flags.DEFINE_string('inDir', set_inDir, "Directory for input numpy files")
flags.DEFINE_string('outDir', set_outDir, "Directory for output numpy files")
flags.DEFINE_string('GoF_test', set_GoF_test, "What test to use: MMD, Sink, or W1")
flags.DEFINE_string('events', "mixed", "Combination of events: mixed, track, or shower")
flags.DEFINE_integer('sinkEps', 1, "Hyperparemeter for SinkHorn Divergence")
flags.DEFINE_integer('bsize', -1, "Override for batch size")

def main(argv): 

	assert FLAGS.GoF_test in ["MMD", "Sink", "W1"]
	assert FLAGS.events in ["mixed", "track", "shower"] 
	
	for compare in ["train", "test"]: 

		print(compare, "samples")

		if compare == 'test': 
			n = 10000
		else: 
			n = 50000 

		# Set batch size  
		if FLAGS.bsize == -1: 
			if FLAGS.GoF_test == "MMD": 
				bSize = 10000
			if FLAGS.GoF_test == "Sink": 
				bSize = 1000 
			if FLAGS.GoF_test == "W1": 
				bSize = 10000
		else: 
			bsize = FLAGS.bsize 

		if FLAGS.GoF_test == 'Sink': 
			outFile = FLAGS.GoF_test+"_"+FLAGS.events+"_"+compare+"_"+str(FLAGS.sinkEps)+".npy"
		else: 
			outFile = FLAGS.GoF_test+"_"+FLAGS.events+"_"+compare+".npy"

		larT = np.load(FLAGS.inDir+"larcv_png_64_"+compare+"_tracks.npy")
		larS = np.load(FLAGS.inDir+"larcv_png_64_"+compare+"_showers.npy")

		# Get selection of LArTPC events 
		if FLAGS.events == 'tracks': 
			lars = larT
		if FLAGS.events == 'showers': 
			lars = larS
		if FLAGS.events == 'mixed': 
			lars = np.concatenate((larT, larS)) 
		if n > lars.shape[0]: 
			n = lars.shape[0]
		idxL = np.random.choice(np.arange(lars.shape[0]), size=n, replace=False) 
		lar = lars[idxL]
		lar = lar.flatten().reshape(n, 64*64)

		print(n, "samples for", outFile) 

		# Tunable hyperparameter for MMD - flags not implemented 
		sigma_list = []
		for i in range(0,6): 
			sigma_list.append(2**(i+10)) 
		#print(sigma_list)

		# Iterate all generated epochs
		epochs = [10, 20, 30, 40, 50, 60, 100, 150, 300] 
		GoF = [] 
		for i, epoch in enumerate(epochs): 
			
			# Open generated epoch 
			genT = np.load(FLAGS.inDir+"gen_epoch"+str(epoch)+"_tracks.npy")
			genS = np.load(FLAGS.inDir+"gen_epoch"+str(epoch)+"_showers.npy")

			# Get selection of generated events  
			if FLAGS.events == 'tracks': 
				gens = genT 
			if FLAGS.events == 'showers': 
				gens = genS
			if FLAGS.events == 'mixed': 
				gens = np.concatenate((genT, genS)) 
			if n > gens.shape[0]: 
				print("Error: n > gens.shape[0]") 
				exit()  
			idxG = np.random.choice(np.arange(gens.shape[0]), size=n, replace=False) 
			gen = gens[idxG] 

			# Reshape generated data to 2D array 
			gen = gen.flatten().reshape(n, 64*64) 

			# Batching 
			score = 0 
			#for b in tqdm(range(n//bSize)): 
			for b in range(n//bSize): 
				larBatch = lar[b*bSize:(b+1)*bSize]
				genBatch = gen[b*bSize:(b+1)*bSize] 
			
				if FLAGS.GoF_test == "Sink": 
					outSinkdiv = two_sample_sinkdiv(larBatch, genBatch, eps=FLAGS.sinkEps) 
					score += outSinkdiv.item() 
				if FLAGS.GoF_test == "MMD": 
					outMMD = MaximumMeanDis_mix(larBatch, genBatch, sigma_list)
					score += outMMD.item() #* np.sqrt(larBatch.shape[0])
				if FLAGS.GoF_test == "W1": 
					score += Wasserstein_1(larBatch, genBatch)

			# Normalize by number of batches 
			score = score / (n//bSize) 

			# Nonfunctioning tests 
			if 0: 
				if FLAGS.GoF_test == "RE": # Not work for single image
					score = RankEnergy(lar, gen) 

				if FLAGS.GoF_test == "SRE": # Not work for single image 
					score = SoftRankEnergy(lar, gen) 

				if FLAGS.GoF_test == "W2": # Very slow, can only handle 100 images 
					score = TwoSampleWTest(lar, gen)[0]

			GoF.append(np.array([score, epoch, gen.shape[0]])) 	

			print("Epoch", epoch, score)

			torch.cuda.empty_cache()
			del genT, genS, gen # Unnecessary 

		np.save(FLAGS.outDir+outFile, GoF) 

if __name__ == '__main__': 
	app.run(main) 


