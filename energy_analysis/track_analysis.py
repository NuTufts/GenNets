from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy import spatial
from scipy import stats
from PIL import Image
import numpy as np
import os

np.set_printoptions(suppress=True) #print numpy in float not scientific notation

## Print image to command line as ASCII  
def ASCII(track): 
	for row in track: 
		for num in row: 
			print(str(int(num)).zfill(2), end='') 
		print()

#fileName = "gen_epoch101_tracks" 
fileName = "Paul_Results_tracks" 

#fileName = "larcv_png_64_test_showers"

tracks = np.load("/home/zimani/energy_analysis/npy_files/"+fileName+".npy")

## How many tracks to iterate 
exitCount = -1

SecondPCA = True

SampleImages = False 

SaveNPY = True

## Initilizations  
lengths = [] 
lengths2 = [] 
angles = []

## Iterate All Tracks 
for counter, track in enumerate(tracks): 

	## Reformate Image to Pixel Location Data Centered at Origin 
	data = np.nonzero(track)
	if len(data[0]) == 0: 
		print("Empty Track at #"+str(counter)) 
		continue
	#data = np.array([data[1], 64-data[0]]) #centered at 32,32
	data = np.array([data[1]-32, 32-data[0]])
	
	# DBSCAN Select Largest Cluster
	db = DBSCAN(eps=3).fit(data.T) 
	largest_cluster = db.labels_ == stats.mode(db.labels_)[0] 
	data = data[:,largest_cluster] 

	## Scikit-Learn Principle Component Analysis 
	pca = PCA(2) 
	try: 
		pca.fit_transform(data.T) 
	except: 
		print("Bad Track at #"+str(counter)) 
		continue 
	
	eigenvector = np.array([pca.components_[0]]).T
	angle = np.arctan2(eigenvector[1],eigenvector[0])[0] 

	## Second Componenet PCA 
	if SecondPCA: 
		eigenvector2 = np.array([pca.components_[1]]).T
		angle2 = np.arctan2(eigenvector2[1],eigenvector2[0])[0] 
		inverse_rotation_matrix2 = np.array([[np.cos(angle2), np.sin(angle2)],
							[-np.sin(angle2), np.cos(angle2)]], dtype="object") 
		data_x2 = np.matmul(inverse_rotation_matrix2, data) 
		length2 = np.rint(np.max(data_x2[0]) - np.min(data_x2[0])) 	
		
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

	if counter == exitCount:
	
		i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
		edge1 = candidates[i] 
		edge2 = candidates[j]

		## Plot Track, PCA Axis, and Convex Hull
		plt.scatter(data[0], data[1], label="Original Track") 
		#plt.scatter(data_x[0], data_x[1], label="Horizontal") 
		plt.plot([0, length*eigenvector[0]], [0, length*eigenvector[1]], c='r', label="PCA Axis 1")
		titleStr = "Length="+str(round(length,4))+", Angle="+str(round(angle,4))+" radians"
		if SecondPCA: 
			plt.plot([0, length2*eigenvector2[0]], [0, length2*eigenvector2[1]], c='g', label="PCA Axis 2")
			titleStr += ", Width="+str(length2)
		plt.scatter([edge1[0], edge2[0]],[ edge1[1], edge2[1]], c='orange')
		plt.plot([edge1[0], edge2[0]],[ edge1[1], edge2[1]], c='orange')
		plt.scatter(0,0, c='black', marker='+') 
		plt.title(titleStr) 
		plt.legend() 
		plt.tight_layout()
		#plt.savefig(str(counter)+"_PCA.png") 
		plt.show() 

		exit() 

	## Save Image For Each Length 	
	if SampleImages and (length not in lengths): 
		im = Image.fromarray(track)
		im = im.convert("L")
		im.save("sample_lengths/len_"+str(length)+"_sample_"+str(counter)+".png") 
		print("Saved Sample Length "+str(length))

	lengths.append(length) 
	angles.append(angle) 

	if SecondPCA: 
		lengths2.append(length2)

if SaveNPY: 
	
	np.save("hists/"+fileName+"_lengths.npy", lengths) 
	np.save("hists/"+fileName+"_angles.npy", angles) 
	
	if SecondPCA: 
		np.save("hists/"+fileName+"_lengths_PCA2.npy", lengths2) 
		print("Saved: "+fileName+" Length, Angle, and Width") 
	else: 
		print("Saved: "+fileName+" Length and Angle") 
		

