from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy import spatial
from scipy import stats
from absl import flags 
from absl import app 
import numpy as np
import os

## Print event to command line as ASCII  
def ASCII(event): 
	np.set_printoptions(suppress=True) #Numpy formatting 
	for row in event: 
		for num in row: 
			print(str(int(num)).zfill(2), end='') 
		print()

set_fileName = "gen_epoch50_tracks"
set_filePath = "/home/zimani/GenNets/npy_files/"
set_outPath = "./hists/"

FLAGS = flags.FLAGS 
flags.DEFINE_bool('saveNPY', True, 'Save track length and width as npy file')
flags.DEFINE_bool('saveAngle', False, 'Save track angle as npy file')
flags.DEFINE_string('fileName', set_fileName, 'Name of track event file, i.e. name_tracks')
flags.DEFINE_string('filePath', set_filePath, 'Directory of track events')
flags.DEFINE_string('outPath', set_outPath, 'Location to save npy files')

def main(argv): 

	# Sanatize Inputs (minimally)
	outPath = FLAGS.outPath 
	if outPath[-1] != "/": 
		outPath += "/" 
	fileName = FLAGS.fileName
	if fileName[-4:-1] == ".np":
		fileName = fileName[:-4] 

	## Initilizations  
	lengths = [] 
	widths = [] 
	angles = []

	tracks = np.load(FLAGS.filePath+FLAGS.fileName+".npy")

	## Iterate All Tracks 
	for counter, track in enumerate(tracks): 

		## Reformate Image to Pixel Location Data Centered at Origin 
		data = np.nonzero(track)
		if len(data[0]) == 0: 
			lengths.append(0) 
			widths.append(0) 
			continue
		#data = np.array([data[1], 64-data[0]]) #centered at 32,32
		data = np.array([data[1]-32, 32-data[0]])
		
		## DBSCAN Select Largest Cluster (Excluding Background Noise) 
		db = DBSCAN(eps=3).fit(data.T) 
		try: 
			largest_cluster = (db.labels_ == stats.mode(db.labels_[db.labels_!=-1]).mode[0]) 
		except: 
			lengths.append(0) 
			widths.append(0) 
			continue
		data = data[:,largest_cluster] 
		
		## Scikit-Learn Principle Component Analysis 
		pca = PCA(2) 
		try: 
			pca.fit_transform(data.T) 
		except: 
			print("Bad Track at #"+str(counter)) 
			length.append(0)
			width.append(0) 
			continue 

		eigenvector = np.array([pca.components_[0]]).T
		angle = np.arctan2(eigenvector[1],eigenvector[0])[0] 

		## Width = PCA 2nd Component  
		eigenvector2 = np.array([pca.components_[1]]).T
		angle2 = np.arctan2(eigenvector2[1],eigenvector2[0])[0] 
		inverse_rotation_matrix2 = np.array([[np.cos(angle2), np.sin(angle2)],
							[-np.sin(angle2), np.cos(angle2)]], dtype="object") 
		data_x2 = np.matmul(inverse_rotation_matrix2, data) 
		width = np.rint(np.max(data_x2[0]) - np.min(data_x2[0])) 	
		width += 1 # minimum width of 1 (not 0) 

		## Track Length 
		pts = np.copy(data.T)
		try: 
			## Convex Hull Length 
			candidates = pts[spatial.ConvexHull(pts).vertices]
			dist_mat = spatial.distance_matrix(candidates, candidates)
			length = np.amax(dist_mat) 
		except: 
			## Straight Line Causes Error in Convex Hull
			## Use Rotate PCA to X-Axis Length Instead   
			inverse_rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
							[-np.sin(angle), np.cos(angle)]], dtype="object") 
			data_x = np.matmul(inverse_rotation_matrix, data) 	
			length = np.rint(np.max(data_x[0]) - np.min(data_x[0])) 	

		# Testing Width  
		if width == -1: 
			print("Counter",counter) 
			print("Width", width) 
			print("Length", length) 
			fig, axs = plt.subplots(1, 2)
			fig.suptitle("Track Width = "+str(width)) 
			ztrack = np.clip(track*1000, 0, 255) 
			axs[0].imshow(track, cmap='gray', interpolation='none')
			axs[0].set_title('Full Track ('+str(np.count_nonzero(track))+')')
			axs[1].scatter(data[0], data[1]) 
			axs[1].set_xlim(-32,32) 
			axs[1].set_ylim(-32,32)
			axs[1].set_aspect('equal')
			axs[1].set_title('Largest Cluster ('+str(len(data[0]))+')')
			plt.tight_layout()
			plt.show()	
			exit() 

		# Testing Width 
		if width == -1: 
			#plt.imshow(track, cmap='gray', interpolation='none') 
			plt.imshow(np.clip(track*10000, 0, 255) , cmap='gray', interpolation='none') 
			plt.scatter(data[0]+32, -data[1]+32, marker=',', alpha=1, color='blue') 
			#plt.title("Track Width = "+str(int(width))) 
			plt.xticks([]) 
			plt.yticks([]) 
			plt.tight_layout()
			#plt.savefig("./widths/width"+str(int(width))+".png") 
			plt.show()
			print("Saved width", str(int(width))) 
			exit()

		## Testing PCA Calculation 
		if counter == -1:
			i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
			edge1 = candidates[i] 
			edge2 = candidates[j]
			## Plot Track, PCA Axis, and Convex Hull
			plt.scatter(data[0], data[1], label="Original Track") 
			#plt.scatter(data_x[0], data_x[1], label="Horizontal") 
			plt.plot([0, length*eigenvector[0]], [0, length*eigenvector[1]], c='r', label="PCA Axis 1")
			titleStr = "Length="+str(round(length,4))+", Angle="+str(round(angle,4))+" radians"
			plt.plot([0, width*eigenvector2[0]], [0, width*eigenvector2[1]], c='g', label="PCA Axis 2")
			titleStr += ", Width="+str(width)
			plt.scatter([edge1[0], edge2[0]],[ edge1[1], edge2[1]], c='orange')
			plt.plot([edge1[0], edge2[0]],[ edge1[1], edge2[1]], c='orange')
			plt.scatter(0,0, c='black', marker='+') 
			plt.title(titleStr) 
			plt.legend() 
			plt.tight_layout()
			#plt.savefig(str(counter)+"_PCA.png") 
			plt.show() 
			exit() 

		lengths.append(length) 
		widths.append(width) 

		if FLAGS.saveAngle: 
			angles.append(angle) 

	if FLAGS.saveNPY: 
		
		np.save(outPath+fileName+"_lengths.npy", lengths)
		np.save(outPath+fileName+"_widths.npy", widths)
		print("Saved", fileName, "Lengths & Widths") 

		if FLAGS.saveAngle: 
			np.save(outPath+fileName+"_angles.npy", angles) 
			print("Saved Angles") 


if __name__ == '__main__': 
	app.run(main)
