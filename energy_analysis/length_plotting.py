import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

## TODO: add flags and streamline 

histPath = "/home/zimani/GenNets/energy_analysis/hists/"

base = np.load(histPath+"larcv_png_64_test_tracks_lengths.npy") 
basePCA2 = np.load(histPath+"larcv_png_64_test_tracks_widths.npy") 
baseName = "LArTPC Tracks Val"

comp10 = np.load(histPath+"gen_epoch10_tracks_lengths.npy") 
comp10PCA2 = np.load(histPath+"gen_epoch10_tracks_widths.npy") 

#comp20 = np.load(histPath+"gen_epoch20_tracks_lengths.npy") 
#comp20PCA2 = np.load(histPath+"gen_epoch20_tracks_lengths_PCA2.npy") 

#comp30 = np.load(histPath+"gen_epoch30_tracks_lengths.npy") 
#comp30PCA2 = np.load(histPath+"gen_epoch30_tracks_lengths_PCA2.npy") 

#comp40 = np.load(histPath+"gen_epoch40_tracks_lengths.npy") 
#comp40PCA2 = np.load(histPath+"gen_epoch40_tracks_lengths_PCA2.npy") 

comp50 = np.load(histPath+"gen_epoch50_tracks_lengths.npy") 
comp50PCA2 = np.load(histPath+"gen_epoch50_tracks_widths.npy") 

#comp100 = np.load(histPath+"gen_epoch100_tracks_lengths.npy") 
#comp100PCA2 = np.load(histPath+"gen_epoch100_tracks_lengths_PCA2.npy") 

comp150 = np.load(histPath+"gen_epoch150_tracks_lengths.npy") 
comp150PCA2 = np.load(histPath+"gen_epoch150_tracks_widths.npy") 

#comp300 = np.load(histPath+"gen_epoch300_tracks_lengths.npy") 
#comp300PCA2 = np.load(histPath+"gen_epoch300_tracks_lengths_PCA2.npy") 

vqvae = np.load(histPath+"VQVAE_tracks_lengths.npy") 
vqvaePCA2 = np.load(histPath+"VQVAE_tracks_widths.npy")

#PCA1 = [comp20, comp30, comp40, comp50, comp100, comp150, comp300]
#PCA2 = [comp20PCA2, comp30PCA2, comp40PCA2, comp50PCA2, comp100PCA2, comp150PCA2, comp300PCA2]
#names = ['10 Epochs', '20 Epochs','30 Epochs', '40 Epochs','50 Epochs', '100 Epochs', '150 Epochs', '300 Epochs']
#colors = ['brown', 'b', 'g', 'r', 'c', 'm', 'y', 'orange'] 

#PCA1 = [comp50, comp100, comp150, comp300] 
#PCA2 = [comp50PCA2, comp100PCA2, comp150PCA2, comp300PCA2] 
#names = ['50 Epochs', '100 Epochs', '150 Epochs', '300 Epochs'] 
#colors = ['c', 'm', 'y', 'orange'] 

PCA1 = [comp10, comp50, comp150] 
PCA2 = [comp10PCA2, comp50PCA2, comp150PCA2] 
names = ['10 Epochs', '50 Epochs', '150 Epochs'] 
styles = ['dashed', '-', 'dotted']
colors = ['g', 'r', 'b'] 

saveFig = True

#histName = "hist_length_compare_all_epochs.png" 
histName = "hist_length_compare_key_epochs.png" 

saveStats = False # Save: [Name, K-S, P-Val, Chi-Square, P-Val]

showVQVAE = False
if showVQVAE: 
	histName = "hist_length_compare_key_vqvae.png" 
	PCA1.append(vqvae) 
	PCA2.append(vqvaePCA2) 
	names.append('VQ-VAE')
	colors.append('y') 
	#styles.append('dashdot') 
	styles.append("-") 

## Figure formatting  
plt.rcParams.update({'font.size': 20})
lWidth = 4

PCAs = [PCA1, PCA2] 
for numPCA, comps in enumerate(PCAs): 

	compStats = [] 	
	plt.figure(figsize=(12,8))

	if numPCA == 0:
		print("Track Length") 
		binwidth = 2 
		xmax = 94
	if numPCA == 1: 
		print()
		print("Track Width") 
		base = basePCA2 
		binwidth = 1
		xmax = 30

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
	countB, _ = np.histogram(np.clip(base, allBins[0], allBins[-1]), bins=allBins)
	
	nB = base.size 

	offB = np.sum(np.logical_or([base<xmin], [xmax<base]))
	offB = nB - np.sum(countB) 
	perOffB = round(100 * offB/nB, 3) 

	print("Name, K-S, P-Val, Chi2, P-Val")

	plt.step(centers, countB/nB, linewidth=lWidth, c="black", label=baseName+" (N="+str(nB)+")")

	## Error Kewords Arguments 
	errgs = dict(ls="none", 
				alpha=0.3,
				elinewidth=None,
				capsize=2) 

	## Plot Errors 
	plt.errorbar(centers-binwidth/2, countB/nB, yerr=np.sqrt(countB)/nB, ecolor="black", **errgs)

	## Comparison Data 
	for i, comp in enumerate(comps): 

		countC, _ = np.histogram(np.clip(comp, allBins[0], allBins[-1]), bins=allBins)

		nC = comp.size 
		offC = np.sum(np.logical_or([comp<xmin], [xmax<comp]))
		perOffC = round(100 * offC/nC, 3) 
		
		# Plot Histogram 
		plt.step(centers, countC/nC, linewidth=lWidth, c=colors[i], linestyle=styles[i], label=names[i]+" (N="+str(nC)+")")

		# Error Bars 
		plt.errorbar(centers-binwidth/2, countC/nC, yerr=np.sqrt(countC)/nC, ecolor=colors[i], **errgs)
		
		# Statistics 
		ks2 = stats.ks_2samp(comp, base)  
		chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
		
		if "VAE" in  names[i]:  
			compStats.append(np.array([names[i], ks2[0], ks2[1], chi[1], chi[0], chi[1]]))
		else: 
			compStats.append(np.array([int(names[i].split(" ")[0]), ks2[0], ks2[1], chi[1], chi[0], chi[1]]))
		print("%s,%.5f,%.6g,%.5f,%.6g" % (names[i], ks2[0], ks2[1], chi[0], chi[1])) 

	if saveStats and numPCA == 0: 
		print(compStats)
		np.save("length_stats.npy", stats) 
	if saveStats and numPCA == 1: 
		np.save("width_stats.npy", stats) 

	## Formatting 
	plt.ylim(bottom=0) 
	plt.xlim(xmin,xmax) 
	plt.ylabel("Fraction of Images")
	plt.legend() 

	# Track Length 
	if numPCA == 0:
		plt.title("Track Length Distribution") 
		plt.xlabel("Length of Track")
		plt.xticks(np.arange(xmin, xmax+1, 10)) 
		plt.ylim(0,0.12)
		plt.tight_layout()
		if saveFig: 
			plt.savefig(histName) 
			print("Saved:", histName) 

	# Track Width
	if numPCA == 1: 
		plt.title("Track Width Distribution") 
		plt.xlabel("Width of Track")
		xTicks = np.arange(xmin, xmax+1, 2).astype('int').astype('str')   
		xTicks[-2] += "+" # Last shown bit as overflow  
		plt.xticks(np.arange(xmin, xmax+1, 2), xTicks) 
		plt.xlim(0, xmax-1.5) 
		plt.ylim(0, 0.28)
		plt.tight_layout()
		if saveFig: 
			histNamePCA2 = histName[:-4]+"_width.png" 
			plt.savefig(histNamePCA2) 
			print("Saved:", histNamePCA2) 

	plt.show()


