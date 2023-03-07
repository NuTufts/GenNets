import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

histPath = "/home/zimani/GenNets/energy_analysis/hists/"

base = np.load(histPath+"larcv_png_64_train_tracks_lengths.npy") 
basePCA2 = np.load(histPath+"larcv_png_64_train_tracks_lengths_PCA2.npy") 
baseName = "LArTPC Tracks"
'''
comp20 = np.load(histPath+"gen_epoch20_tracks_lengths.npy") 
comp20PCA2 = np.load(histPath+"gen_epoch20_tracks_lengths_PCA2.npy") 

comp30 = np.load(histPath+"gen_epoch30_tracks_lengths.npy") 
comp30PCA2 = np.load(histPath+"gen_epoch30_tracks_lengths_PCA2.npy") 

comp40 = np.load(histPath+"gen_epoch40_tracks_lengths.npy") 
comp40PCA2 = np.load(histPath+"gen_epoch40_tracks_lengths_PCA2.npy") 

comp50 = np.load(histPath+"gen_epoch50_tracks_lengths.npy") 
comp50PCA2 = np.load(histPath+"gen_epoch50_tracks_lengths_PCA2.npy") 
'''
comp100 = np.load(histPath+"gen_epoch100_tracks_lengths.npy") 
comp100PCA2 = np.load(histPath+"gen_epoch100_tracks_lengths_PCA2.npy") 
'''
comp150 = np.load(histPath+"gen_epoch150_tracks_lengths.npy") 
comp150PCA2 = np.load(histPath+"gen_epoch150_tracks_lengths_PCA2.npy") 

comp300 = np.load(histPath+"gen_epoch300_tracks_lengths.npy") 
comp300PCA2 = np.load(histPath+"gen_epoch300_tracks_lengths_PCA2.npy") 

vqvae = np.load(histPath+"Paul_Results_tracks_lengths.npy") 
vqvaePCA2 = np.load(histPath+"Paul_Results_tracks_lengths_PCA2.npy")
'''
comp100_v2 = np.load(histPath+"gen_epoch100_v2_tracks_lengths.npy") 
comp100PCA2_v2 = np.load(histPath+"gen_epoch100_v2_tracks_lengths_PCA2.npy") 

comp100a_v2 = np.load(histPath+"gen_epoch100a_v2_tracks_lengths.npy") 
comp100PCA2a_v2 = np.load(histPath+"gen_epoch100a_v2_tracks_lengths_PCA2.npy") 


#PCA1 = [comp20, comp30, comp40, comp50, comp100, comp150]
#PCA2 = [comp20PCA2, comp30PCA2, comp40PCA2, comp50PCA2, comp100PCA2, comp150PCA2]
#names = ['20 Epochs','30 Epochs', '40 Epochs','50 Epochs', '100 Epochs', '150 Epochs']
#colors = ['b', 'g', 'r', 'c', 'm', 'y'] 
'''
PCA1 = [comp50, comp100, comp150, comp300] 
PCA2 = [comp50PCA2, comp100PCA2, comp150PCA2, comp300PCA2] 
names = ['50 Epochs', '100 Epochs', '150 Epochs', '300 Epochs'] 
colors = ['c', 'm', 'y', 'orange'] 
'''
PCA1 = [comp100, comp100_v2, comp100a_v2] 
PCA2 = [comp100PCA2, comp100PCA2_v2, comp100PCA2a_v2] 
names = ['100 Epochs v1', '100 Epochs v2a', '100b Epochs v2b'] 
colors = ['c', 'y', 'r'] 


saveFig = True 

#histName = "hist_length_compare_all_epochs.png" 
histName = "hist_length_compare_100s_epochs.png" 

statsCSV = True # Print errors: Name, K-S, P-Val, Chi-Square, P-Val

showVQVAE = False  
if showVQVAE: 
	histName = "hist_length_compare_all_vqvae.png" 
	PCA1.append(vqvae) 
	PCA2.append(vqvaePCA2) 
	names.append('VQ-VAE')
	colors.append('magenta') 

#comp1 = np.load(histPath+"larcv_png_64_train_showers_lengths.npy") 
#comp1PCA2 = np.load(histPath+"larcv_png_64_train_showers_lengths_PCA2.npy") 

#PCA1 = [comp1] 
#PCA2 = [comp1PCA2] 
#colors = ['blue']
#names = ['LArTPC Showers']


PCAs = [PCA1, PCA2] 
for numPCA, comps in enumerate(PCAs): 

	if numPCA == 0:
		print("Track Length") 
		binwidth = 2 
		xmax = 100 
	if numPCA == 1: 
		print()
		print("Track Width") 
		base = basePCA2 
		binwidth = 1
		xmax = 35
		#xmax = 90

	xmin = 0
	xmax = (binwidth * np.ceil(xmax/binwidth)) #round to multiple  
	allBins = np.arange(-binwidth,xmax,binwidth)
	centers = 0.5*(allBins[1:] + allBins[:-1])  


	## General Track Info 
	mins = [np.min(base)] 
	maxs = [np.max(base)] 
	for comp in comps: 
		mins.append(np.min(comp)) 
		maxs.append(np.max(comp)) 

	## Baseline Bin Data 
	countB, _ = np.histogram(base, bins=allBins) 
	
	nB = base.size 

	offB = np.sum(np.logical_or([base<xmin], [xmax<base]))
	offB = nB - np.sum(countB) 
	perOffB = round(100 * offB/nB, 3) 
	
	if not statsCSV:  
		print("---")
		print("Shortest Track = %0.3f" % np.min(mins))
		print("Longest Track = %0.3f" % np.max(maxs))
		print("Bins 0:%d | Bin Width = %d" % (xmax, binwidth)) 
		print("---") 
		print() 
		print(baseName) 
		print("Tracks not shown =", str(offB), "("+str(perOffB)+"%)") 
		print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countB))) 
	else: 
		print("Name, K-S, P-Val, Chi2, P-Val") 

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
		offC = np.sum(np.logical_or([comp<xmin], [xmax<comp]))
		perOffC = round(100 * offC/nC, 3) 
		
		# Plot Histogram 
		plt.step(centers, countC/nC, linewidth=2, c=colors[i], label=names[i]+" (N="+str(nC)+")")

		# Error Bars 
		plt.errorbar(centers-binwidth/2, countC/nC, yerr=np.sqrt(countC)/nC, ecolor=colors[i], **errgs)
		
		# Statistics 
		ks2 = stats.ks_2samp(comp, base)  
		chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
		
		if not statsCSV: 
			print()
			print(names[i]) 
			#print("Tracks not shown =", str(offC), "("+str(perOffC)+"%)") 
			print("K-S Test = %.5f" % ks2[0]) 
			print("   P = %.6g" % ks2[1]) 
			print("Chi-square = %.5f" % chi[0]) 
			print("   P = %.6g" % chi[1]) 
		else: 	
			print("%s,%.5f,%.6g,%.5f,%.6g" % (names[i], ks2[0], ks2[1], chi[0], chi[1])) 

	## Formatting 
	plt.ylim(bottom=0) 
	plt.xlim(xmin,xmax) 
	plt.ylabel("Fraction of Images")
	plt.legend() 

	if numPCA == 0: # First PCA
		plt.title("Track Length Distribution") 
		plt.xlabel("Length of Track")
		plt.tight_layout()
		if saveFig: 
			plt.savefig(histName) 
			print("Saved:", histName) 

	if numPCA == 1: # Second PCA
		plt.title("Track 2nd PCA Length") 
		plt.xlabel("Width of Track")
		plt.tight_layout()
		if saveFig: 
			histNamePCA2 = histName[:-4]+"_PCA2.png" 
			plt.savefig(histNamePCA2) 
			print("Saved:", histNamePCA2) 

	plt.show()

