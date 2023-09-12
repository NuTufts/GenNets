#import matplotlib as mpl 
#mpl.use('Agg') 
import matplotlib.pyplot as plt
from absl import flags 
from absl import app
import itertools 
import numpy as np
import cv2 

set_eventNum = 0 
set_eventType = "track" 
set_distMode = 'l2'
set_inDir = "/home/zimani/GenNets/npy_files/" 
set_genFile = "gen_epoch50" 
set_larFile = "larcv_png_64_train" 

FLAGS = flags.FLAGS 
flags.DEFINE_integer('eventNum', set_eventNum, 'Index of generated event to find neighbors for')
flags.DEFINE_bool('select', False, 'Show 16 images, useful for choosing eventNum')
flags.DEFINE_string('eventType', set_eventType, 'Track or shower ')
flags.DEFINE_string('distMode', set_distMode, 'EMD or l2 for distance')
flags.DEFINE_string('inDir', set_inDir, 'Directory location of numpy files')
flags.DEFINE_bool('near', True, 'Nearest or farthest neighbors')
flags.DEFINE_integer('numNeighbors', 5, 'Number of nearest neighbors to display')
flags.DEFINE_string('genFile', set_genFile, 'Name of generated file without _tracks(showers)')
flags.DEFINE_string('larFile', set_larFile, 'Name of comparison file without _tracks(showers)')
flags.DEFINE_bool('showPlot', True, 'Show plot')
flags.DEFINE_bool('saveNPY', False, 'Save distances as npy file')

def main(argv): 

	assert FLAGS.eventType in ["track", "shower"]
	assert FLAGS.distMode in ["EMD", "l2"]

	gens = np.load(FLAGS.inDir+FLAGS.genFile+"_"+FLAGS.eventType+"s.npy") 

	lars = np.load(FLAGS.inDir+FLAGS.larFile+"_"+FLAGS.eventType+"s.npy") 

	img = gens[FLAGS.eventNum,:] 
	if FLAGS.select: 
		fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
		for i, ax in enumerate(axes.flatten()): 
			ax.imshow(gens[FLAGS.eventNum+i,:], cmap='gray', interpolation='none')
			ax.set_title(str(FLAGS.eventNum+i))
			ax.axis('off') 
		plt.tight_layout() 
		plt.show() 
		exit() 

	if FLAGS.distMode == 'EMD':
		# OpenCV EMD setup: signature matrix 
		xx, yy = np.meshgrid(np.arange(64), np.arange(64))
		xx = xx.ravel()
		yy = yy.ravel()
		img = img / np.sum(img) 
		imgSig = np.vstack((img.ravel(), xx, yy)).T.astype(np.float32)

	if FLAGS.near: 
		nn = -1 
	else: 
		nn = 1 

	# Initializations
	larDists = [] 
	genDists = [] 

	# Compare to training data 
	for lar in lars: 
		if FLAGS.distMode == 'l2': 
			larDist = int(np.linalg.norm(img - lar))
		elif FLAGS.distMode == 'EMD': 
			lar = lar / np.sum(lar) 
			larSig = np.vstack((lar.ravel(), xx, yy)).T.astype(np.float32)
			larDist  = cv2.EMD(imgSig, larSig, cv2.DIST_L2)[0]
		else: 
			print("Error: invalid distMode") 
			exit() 
		larDists.append(larDist) 
	larDists = np.asarray(larDists)
	print("Training Neighbors Found")

	if FLAGS.saveNPY: 
		np.save(FLAGS.distMode+"_dists_lar_"+FLAGS.eventType+"_"+str(FLAGS.eventNum)+".npy", larDists) 

	# Sort nearest neighbors 
	idxSortL = (nn*larDists).argsort() 
	larDists = larDists[idxSortL[::-1]] 
	larSort = lars[idxSortL[::-1]] 

	# Compare to generated data
	for gen in gens: 
		if FLAGS.distMode == 'l2': 
			genDist = int(np.linalg.norm(img - gen))
		elif FLAGS.distMode == 'EMD': 
			gen = gen / np.sum(gen) 
			genSig = np.vstack((gen.ravel(), xx, yy)).T.astype(np.float32)
			genDist = cv2.EMD(imgSig, genSig, cv2.DIST_L2)[0]
		genDists.append(genDist) 
	genDists = np.asarray(genDists) 
	print("Generated Neighbors Found")

	if FLAGS.saveNPY: 
		np.save(FLAGS.distMode+"_dists_gen_"+FLAGS.eventType+"_"+str(FLAGS.eventNum)+".npy", genDists) 

	# Sort nearest neighbors 
	idxSortG = (nn*genDists).argsort() 
	genDists = genDists[idxSortG[::-1]] 
	genSort = gens[idxSortG[::-1]] 

	# Remove self match
	if FLAGS.near: 
		genDists = genDists[1:] 
		genSort = genSort[1:] 

	## Plotting 

	plt.figure(figsize=(14,4)) 

	plt.subplot2grid(shape=(2, 7), loc=(0,0), colspan=2, rowspan=2) 
	plt.imshow(img, cmap='gray', interpolation='nearest') 
	plt.title("Generated") 
	plt.axis('off') 

	for i in range(FLAGS.numNeighbors): 
		plt.subplot2grid((2,7), (0,i+2)) 
		plt.imshow(larSort[i], cmap='gray', interpolation='nearest')
		plt.title(larDists[i]) 
		plt.axis('off') 

		plt.subplot2grid((2,7), (1,i+2)) 
		plt.imshow(genSort[i], cmap='gray', interpolation='nearest')
		plt.title(genDists[i]) 
		plt.axis('off') 

	plt.tight_layout() 

	plt.savefig(FLAGS.distMode+"_neighbors_"+FLAGS.eventType+"_"+str(FLAGS.eventNum)+".png")
	print("Saved", FLAGS.distMode+"_neighbors_"+FLAGS.eventType+"_"+str(FLAGS.eventNum)+".png") 

	if FLAGS.showPlot: 
		plt.show() 

if __name__ == '__main__': 
	app.run(main) 
