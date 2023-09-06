import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

##############
# Deprecated #
##############

histPath = "/home/zimani/energy_analysis/hists/"

#wavelet = "db1" 
#wavelet = "haar" 
#wavelet = "sums"
wavelet = "thresh_norm" 
#wavelet = "energy" 
#wavelet = "thresh"

baseNPY = np.load(histPath+"larcv_png_64_train_showers_wavelets_"+wavelet+".npy") 
comp1NPY = np.load(histPath+"larcv_png_64_train_tracks_wavelets_"+wavelet+".npy") 

#comp2NPY = np.load(histPath+"gen_epoch25_showers_wavelets_"+wavelet+".npy") 
#comp2NPY = np.load(histPath+"gen_epoch50_showers_wavelets_"+wavelet+".npy") 
#comp3NPY = np.load(histPath+"gen_epoch100_showers_wavelets_"+wavelet+".npy") 
#comp4NPY = np.load(histPath+"gen_epoch150_showers_wavelets_"+wavelet+".npy") 

baseName = "LArTPC Showers"
names = ["LArTPC Tracks"] 
colors = ["blue"] 
#names = ["LarTPC Tracks", "50 Epochs", "100 Epochs", "150 Epochs"]
#colors = ["blue", "orange", "red", "green"]
waveName = ["Approximate", "Horizontal", "Vertical", "Diagonal"] 
waveName = ["H+V"]

saveFig = True 

#for waveNum in range(1, 5): 
for waveNum in range(1, 2): 

	base = baseNPY[:,waveNum-1]  
	comp1 = comp1NPY[:,waveNum-1] 
	#comp2 = comp2NPY[:,waveNum-1] 
	#comp3 = comp3NPY[:,waveNum-1] 
	#comp4 = comp4NPY[:,waveNum-1] 

	base = baseNPY[:,1] + baseNPY[:,2]# + baseNPY[:,3]
	comp1 = comp1NPY[:,1] + comp1NPY[:,2]# + comp1NPY[:,3]
	#comp2 = comp2NPY[:,1] + comp2NPY[:,2] + comp2NPY[:,3]
	#comp3 = comp3NPY[:,1] + comp3NPY[:,2] + comp3NPY[:,3]
	
	#comps = [comp1, comp2, comp3, comp4] 
	comps = [comp1] 

	if wavelet == "sums": 
		if waveNum == 1: 
			binwidth = 200
			xmin = 500
			xmax = 17500

		if waveNum >= 2: 
			binwidth = 50
			xmin = -1500
			xmax = 1500
	
	if wavelet == "thresh": 
		binwidth = 2
		xmin = -binwidth 
		if waveNum == 1:
			xmax = 150 
		else: 
			xmax = 80

	if wavelet == "thresh_norm": 
		binwidth = 0.01
		xmin = -3*binwidth 
		xmax = 0.5 + 3*binwidth 
	
	if wavelet == "energy": 
		binwidth = 40000
		xmin = -binwidth 
		if waveNum == 1:
			xmax = 3000000
		else: 
			xmax = 1000000

	if wavelet == "haar": 
		if waveNum == 1: 
			binwidth = 20
			xmin = -binwidth
			xmax = 640

		if waveNum == 2: 
			binwidth = 4
			xmin = -88
			xmax = 88

		if waveNum == 3: 
			binwidth = 4
			xmin = -88
			xmax = 88

		if waveNum == 4: 
			binwidth = 4
			xmin = -72
			xmax = 72


	if wavelet == 'db1': 
		if waveNum == 1: 
			binwidth = 20
			xmin = -20
			xmax = 600

		if waveNum == 2: 
			binwidth = 20
			xmin = -350
			xmax = 350

		if waveNum == 3: 
			binwidth = 20
			xmin = -350
			xmax = 350

		if waveNum == 4: 
			binwidth = 20
			xmin = -450
			xmax = 450

	xmax = (binwidth * np.ceil(xmax/binwidth)) #round to multiple  
	allBins = np.arange(xmin,xmax,binwidth)
	centers = 0.5*(allBins[1:] + allBins[:-1])  

	## General Data Info
	mins = [np.min(base)]
	maxs = [np.max(base)]
	for comp in comps:
		mins.append(np.min(comp))
		maxs.append(np.max(comp))
	print("---")
	print("Wave Number =", str(waveNum)) 
	print("Shortest Track = %0.3f" % np.min(mins))
	print("Longest Track = %0.3f" % np.max(maxs))
	print("Bins 0:%d | Bin Width = %d" % (xmax, binwidth))
	print("")
 
	## Baseline Bin Data
	countB, _ = np.histogram(base, bins=allBins)

	nB = base.size
	offB = np.sum(np.logical_or([base<xmin], [xmax<base]))
	offB = nB - np.sum(countB)
	perOffB = round(100 * offB/nB, 3)

	print(baseName)
	print("Tracks not shown =", str(offB), "("+str(perOffB)+"%)")
	print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countB)))

	plt.step(centers, countB/nB, linewidth=2, c="black", label=baseName+" (N="+str(nB)+")")

	## Error Kewords Arguments 
	errgs = dict(ls="none", 
				alpha=0.3,
				elinewidth=None,
				capsize=2) 

	## Plot Errors
	plt.errorbar(centers-binwidth/2, countB/nB, yerr=np.sqrt(countB)/nB, ecolor="black", **errgs)


	## Comparison Data 
	for i, comp in enumerate(comps):

		countC, _ = np.histogram(comp, bins=allBins)

		nC = comp.size
		#offC = nC - np.sum(countC)
		offC = np.sum(np.logical_or([comp<xmin], [xmax<comp]))
		perOffC = round(100 * offC/nC, 3)

		print()
		print(names[i])
		print("Tracks not shown =", str(offC), "("+str(perOffC)+"%)")

		# Plot Histogram
		plt.step(centers, countC/nC, linewidth=2, c=colors[i], label=names[i]+" (N="+str(nC)+")")

		# Error Bars
		plt.errorbar(centers-binwidth/2, countC/nC, yerr=np.sqrt(countC)/nC, ecolor=colors[i], **errgs)

		# Statistics
		#ks2 = stats.ks_2samp(comp, base)
		#print("K-S Test = %.5f" % ks2[0])
		#print("   P = %.6g" % ks2[1])

		#chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
		#print("Chi-square = %.5f" % chi[0])
		#print("   P = %.6g" % chi[1])

	## Formatting 
	titleText = "Depth 1: "+waveName[waveNum-1]
	if wavelet == "energy": 
		#plt.title(titleText+" Energy") 
		#plt.title("Energy: $|H|^2+|V|^2+|D|^2$") 
		plt.title("Energy: $|A|^2$") 
	if wavelet == "thresh_norm": 
		plt.title(titleText+" Threshold 50 Count Normalized") 
	if wavelet == 'sums': 
		plt.title(titleText+" HVD Sum") 
	if wavelet == 'haar': 
		plt.title("Shower Wavelet Max Depth "+waveName[waveNum-1]) 
	if wavelet == 'db1': 
		plt.title("Shower Wavelet Coefficient #"+str(waveNum)) 

	plt.ylim(bottom=0) 
	plt.xlim(xmin,xmax) 
	plt.xlabel("Fraction of Pixels Above Threshold")
	#plt.xlabel("Energy")
	plt.ylabel("Fraction of Images")
	plt.legend() 
	plt.tight_layout()
	if saveFig: 
		plt.savefig("wavelet_"+wavelet+"_coeff"+str(waveNum)+"_track_shower_HV.png") 
		#plt.savefig("wavelet_"+wavelet+"_coeff"+str(waveNum)+"_gens.png") 
		#plt.savefig("wavelet_"+wavelet+"_coeff"+str(waveNum)+"_all.png") 
	plt.show()


