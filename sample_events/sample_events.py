import matplotlib.pyplot as plt 
from absl import flags 
from absl import app
import numpy as np


set_inDir = "/home/zimani/GenNets/npy_files/"
set_genFile = "gen_epoch50" 
set_larFile = "larcv_png_64_train" 
set_pltTitle = "Generated Images (Epoch 50)"

FLAGS = flags.FLAGS 
flags.DEFINE_bool('gen', True, 'Sample from generated images (True) or lar files (False)')
flags.DEFINE_string('inDir', set_inDir, "Directory for input numpy files")
flags.DEFINE_string('genFile', set_genFile, 'Name of generated file without _tracks(showers)')
flags.DEFINE_string('larFile', set_larFile, 'Name of comparison file without _tracks(showers)')
flags.DEFINE_bool('showPlot', True, 'Show plot')
flags.DEFINE_string('pltTitle', set_pltTitle, 'Plot title')
flags.DEFINE_integer('numRows', 10, 'Number of rows')
flags.DEFINE_integer('numCols', 7, 'Number of columns')

def main(argv): 

	if FLAGS.gen: 
		genTracks = np.load(FLAGS.inDir+"gen_epoch50_tracks.npy") 
		genShowers = np.load(FLAGS.inDir+"gen_epoch50_showers.npy") 
		events = np.vstack((genTracks, genShowers))
		pltTitle = FLAGS.pltTitle
		figName = "gen"
	else: 
		larcvTracks = np.load(FLAGS.inDir+"larcv_png_64_train_tracks.npy")
		larcvShowers = np.load(FLAGS.inDir+"larcv_png_64_train_showers.npy")
		events = np.vstack((larcvTracks, larcvShowers)) 
		if "Generated" in FLAGS.pltTitle: 
			pltTitle = "Training Images"
		else: 
			pltTitle = FLAGS.pltTitle
		figName = 'larcv'

	idx = np.random.choice(np.arange(events.shape[0]), size=FLAGS.numRows*FLAGS.numCols, replace=False)
	samples = events[idx]  

	plt.figure(figsize=(8, 12)) 
	for i in range(FLAGS.numRows*FLAGS.numCols): 
		plt.subplot(FLAGS.numRows, FLAGS.numCols, i+1) 
		plt.imshow(samples[i], cmap='gray', interpolation='none') 
		plt.axis('off') 

	plt.suptitle(pltTitle, fontsize=12) 
	plt.tight_layout() 

	plt.savefig(figName+"_samples.png") 

	if FLAGS.showPlot: 
		plt.show() 

if __name__ == '__main__': 
	app.run(main) 


