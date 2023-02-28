import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

histPath = "/home/zimani/energy_analysis/hists/"

#baseT = np.load(histPath+"larcv_png_64_train_tracks_lengths_PCA2.npy") 
baseT = np.load(histPath+"larcv_png_64_train_tracks_lengths.npy") 
#baseS = np.load(histPath+"larcv_png_64_train_showers_lengths_PCA2.npy") 
baseS = np.load(histPath+"larcv_png_64_train_showers_lengths.npy") 
baseName = "LArTPC"

comp20T = np.load(histPath+"gen_epoch20_tracks_lengths_PCA2.npy") 
comp20S = np.load(histPath+"gen_epoch20_showers_lengths_PCA2.npy") 

comp30T = np.load(histPath+"gen_epoch30_tracks_lengths_PCA2.npy") 
comp30S = np.load(histPath+"gen_epoch30_showers_lengths_PCA2.npy") 

comp40T = np.load(histPath+"gen_epoch40_tracks_lengths_PCA2.npy") 
comp40S = np.load(histPath+"gen_epoch40_showers_lengths_PCA2.npy") 

comp50T = np.load(histPath+"gen_epoch50_tracks_lengths_PCA2.npy") 
comp50S = np.load(histPath+"gen_epoch50_showers_lengths_PCA2.npy") 

comp100T = np.load(histPath+"gen_epoch100_tracks_lengths_PCA2.npy") 
comp100S = np.load(histPath+"gen_epoch100_showers_lengths_PCA2.npy") 

comp150T = np.load(histPath+"gen_epoch150_tracks_lengths_PCA2.npy") 
comp150S = np.load(histPath+"gen_epoch150_showers_lengths_PCA2.npy") 

paulT = np.load(histPath+"Paul_Results_tracks_lengths_PCA2.npy")
paulS = np.load(histPath+"Paul_Results_showers_lengths_PCA2.npy")

PCAT = [comp20T, comp30T, comp40T, comp50T, comp100T, comp150T, paulT] 
PCAS = [comp20S, comp30S, comp40S, comp50S, comp100S, comp150S, paulS] 

PCAT = []
PCAS = []

twoComp = True 
if twoComp: 
	#hist_suffix = "showers_charges" 
	#hist_suffix = "tracks_lengths" 
	hist_suffix = "tracks_lengths_PCA2" 
	baseT = np.load(histPath+"larcv_png_64_train_"+hist_suffix+".npy") 
	baseS = np.load(histPath+"larcv_png_64_test_"+hist_suffix+".npy") 
	baseZ = np.load(histPath+"gen_epoch100_"+hist_suffix+".npy") 
	baseName = "100 Epochs"

names = ['20 Epochs','30 Epochs', '40 Epochs','50 Epochs', '100 Epochs', '150 Epochs', 'VQ-VAE']
colors = ['b', 'g', 'r', 'orange','c', 'm', 'y',] 

saveFig = True 
histName = "hist_width_compare_validation.png" 

if 1:  
	
	if "charge" in hist_suffix: 
		binwidth = 1000
		xmax = 41000
	if "length." in hist_suffix:  
		binwidth = 2
		xmax = 100
	if "PCA2" in hist_suffix: 
		binwidth = 2 
		xmax = 50 
	xmin = 0
	xmax = (binwidth * np.ceil(xmax/binwidth)) #round to multiple  
	allBins = np.arange(-binwidth,xmax,binwidth)
	centers = 0.5*(allBins[1:] + allBins[:-1])  
	
	if 0: 
		## General Track Info 
		mins = [np.min(base)] 
		maxs = [np.max(base)] 
		for comp in comps: 
			mins.append(np.min(comp)) 
			maxs.append(np.max(comp)) 
		print("---")
		print("Shortest Track = %0.3f" % np.min(mins))
		print("Longest Track = %0.3f" % np.max(maxs))
		print("Bins 0:%d | Bin Width = %d" % (xmax, binwidth)) 
		print("---") 

	## Baseline Bin Data 
	countBT, _ = np.histogram(baseT, bins=allBins) 
	countBS, _ = np.histogram(baseS, bins=allBins) 
	countBZ, _ = np.histogram(baseZ, bins=allBins) 
	
	nBT = baseT.size 
	offBT = np.sum(np.logical_or([baseT<xmin], [xmax<baseT]))
	offBT = nBT - np.sum(countBT) 
	perOffBT = round(100 * offBT/nBT, 3) 

	nBS = baseS.size 
	offBS = np.sum(np.logical_or([baseS<xmin], [xmax<baseS]))
	offBS = nBS - np.sum(countBS) 
	perOffBS = round(100 * offBS/nBS, 3) 

	nBZ = baseZ.size 
	offBZ = np.sum(np.logical_or([baseZ<xmin], [xmax<baseZ]))
	offBZ = nBZ - np.sum(countBZ) 
	perOffBZ = round(100 * offBZ/nBZ, 3) 


	print(baseName, "Tracks") 
	print("Tracks not shown =", str(offBT), "("+str(perOffBT)+"%)") 
	#print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countBT))) 

	print(baseName, "Showers") 
	print("Tracks not shown =", str(offBS), "("+str(perOffBS)+"%)") 
	#print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countBS))) 
	
	print(baseName, "Showers") 
	print("Tracks not shown =", str(offBZ), "("+str(perOffBZ)+"%)") 

	if twoComp: 
		plt.step(centers, countBT/nBT, linewidth=2, c="black", label="LArTPC Training (N="+str(nBT)+")")
		plt.step(centers, countBS/nBS, linewidth=2, c="blue", label="LArTPC Validation (N="+str(nBS)+")")
		plt.step(centers, countBZ/nBZ, linewidth=2, c="red", label="Gen 100 Epochs (N="+str(nBZ)+")")
	else: 	
		plt.step(centers, countBT/nBT, linewidth=2, c="black", label=baseName+" Track (N="+str(nBT)+")")
		plt.step(centers, countBS/nBS, linewidth=2, linestyle='--', c="black", label=baseName+" Shower (N="+str(nBS)+")")

	## Error Kewords Arguments 
	errgs = dict(ls="none", 
				alpha=0.3,
				elinewidth=None,
				capsize=2) 

	## Plot Errors 
	plt.errorbar(centers-binwidth/2, countBT/nBT, yerr=np.sqrt(countBT)/nBT, ecolor="black", **errgs)
	plt.errorbar(centers-binwidth/2, countBS/nBS, yerr=np.sqrt(countBS)/nBS, ecolor="blue", **errgs)
	plt.errorbar(centers-binwidth/2, countBZ/nBZ, yerr=np.sqrt(countBZ)/nBZ, ecolor="red", **errgs)

	## Comparison Data 
	for i, (compT, compS) in enumerate(zip(PCAT, PCAS)): 

		countT, _ = np.histogram(compT, bins=allBins) 
		countS, _ = np.histogram(compS, bins=allBins) 
		
		nT = compT.size 
		offT = np.sum(np.logical_or([compT<xmin], [xmax<compT]))
		perOffT = round(100 * offT/nT, 3) 

		nS = compS.size 
		offS = np.sum(np.logical_or([compS<xmin], [xmax<compS]))
		perOffS = round(100 * offS/nS, 3) 
		
		#print() 
		#print(names[i])  
	#	print("Tracks not shown =", str(offC), "("+str(perOffC)+"%)") 
		
		# Plot Histogram 
		plt.step(centers, countT/nT, linewidth=2, c=colors[i], label=names[i]+" Tracks (N="+str(nT)+")")
		plt.step(centers, countS/nS, linewidth=2, c=colors[i], linestyle='--', label=names[i]+" Showers (N="+str(nS)+")")

		# Error Bars 
		plt.errorbar(centers-binwidth/2, countT/nT, yerr=np.sqrt(countT)/nT, ecolor=colors[i], **errgs)
		plt.errorbar(centers-binwidth/2, countS/nS, yerr=np.sqrt(countS)/nS, ecolor=colors[i], **errgs)
		
		# Statistics 
		#ks2 = stats.ks_2samp(comp, base)  
		#print("K-S Test = %.5f" % ks2[0]) 
		#print("   P = %.6g" % ks2[1]) 

		#chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
		#print("Chi-square = %.5f" % chi[0]) 
		#print("   P = %.6g" % chi[1]) 

		
	print() 

	## Formatting 
	plt.ylim(bottom=0) 
	plt.xlim(xmin,xmax) 
	plt.ylabel("Fraction of Images")
	
	if "charge" in hist_suffix: 
		plt.title("Charge Comparison") 
		plt.xlabel("Charge Deposited") 
	elif "PCA" in hist_suffix: 
		plt.title("Width Comparison") 
		plt.xlabel("Width of Event") 
	else: 
		plt.title("Length Comparison")  
		plt.xlabel("Length of Event") 

	plt.tight_layout()
	plt.legend() 
	
	if saveFig: 
		plt.savefig(histName) 

	plt.show() 



