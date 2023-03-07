import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

histPath = "/home/zimani/GenNets/energy_analysis/hists/"

base = np.load(histPath+"larcv_png_64_train_showers_charges.npy") 
baseName = "LArTPC" 

#comp20 = np.load(histPath+"gen_epoch20_showers_charges.npy") 
#comp30 = np.load(histPath+"gen_epoch30_showers_charges.npy") 
#comp40 = np.load(histPath+"gen_epoch40_showers_charges.npy") 
#comp50 = np.load(histPath+"gen_epoch50_showers_charges.npy") 
comp100 = np.load(histPath+"gen_epoch100_showers_charges.npy") 
comp100_v2 = np.load(histPath+"gen_epoch100_v2_showers_charges.npy") 
comp100a_v2  = np.load(histPath+"gen_epoch100a_v2_showers_charges.npy") 
#comp150 = np.load(histPath+"gen_epoch150_showers_charges.npy") 
#comp300 = np.load(histPath+"gen_epoch300_showers_charges.npy") 
#vqvae = np.load(histPath+"Paul_Results_showers_charges.npy") 

#comps = [comp20, comp30, comp40, comp50, comp100, comp150] 
#names = ['20 Epochs', '30 Epochs', '40 Epochs','50 Epochs', '100 Epochs', '150 Epochs']
#colors = ['b', 'g', 'r', 'c', 'm', 'y'] 

#comps = [comp50, comp100, comp150, comp300] 
#names = ['50 Epochs', '100 Epochs', '150 Epochs', '300 Epochs'] 
#colors = ['c', 'm', 'y', 'orange'] 

comps = [comp100, comp100_v2, comp100a_v2] 
names = ['100 Epochs v1', '100 Epochs v2a', '100b Epochs v2b'] 
colors = ['c', 'y', 'r'] 

saveFig = True 
#histName = "hist_charge_compare_all_epochs.png" 
histName = "hist_charge_compare_100s_epochs.png" 

statsCSV = True # Print errors: Name, K-S, P-Val, Chi-Square, P-Val

showVQVAE = False
if showVQVAE: 
	histName = "hist_charge_compare_all_vqvae.png" 
	comps.append(vqvae) 
	names.append('VQ-VAE')

binwidth = 1000
xmax = 41000
xmin = 0 
xmax = (binwidth * np.ceil(xmax/binwidth)) #round to multiple  
allBins = np.arange(-binwidth,xmax,binwidth)
centers = 0.5*(allBins[1:] + allBins[:-1])  

## General Data Info
mins = [np.min(base)]
maxs = [np.max(base)]
for comp in comps:
	mins.append(np.min(comp))
	maxs.append(np.max(comp))

## Baseline Bin Data
countB, _ = np.histogram(base, bins=allBins)

nB = base.size
offB = nB - np.sum(countB)
perOffB = round(100 * offB/nB, 3)

if not statsCSV:
	print("---")
	print("Least Charge = %0.3f" % np.min(mins))
	print("Most Charge = %0.3f" % np.max(maxs))
	print("Bins 0:%d | Bin Width = %d" % (xmax, binwidth))
	print("---")
	print()
	print(baseName)
	print("Tracks not shown =", str(offB), "("+str(perOffB)+"%)")
	print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countB)))
else:
	print("Shower Charge")
	print("Name, K-S, P-Val, Chi2, P-Val")


plt.step(centers, countB/nB, linewidth=2, c="black", label=baseName+" (N="+str(nB)+")")

## Error Kewords Arguments
errgs = dict(ls="none",
	        alpha=0.3,
	        elinewidth=9.3,
	        capsize=0)

## Plot Errors
plt.errorbar(centers-binwidth/2, countB/nB, yerr=np.sqrt(countB)/nB, ecolor="black", **errgs)

## Comparison Data
for i, comp in enumerate(comps):

	countC, _ = np.histogram(comp, bins=allBins)

	nC = comp.size
	offC = nC - np.sum(countC)
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
#plt.ylim(bottom=0) 
if showVQVAE: 
	plt.ylim(0, 0.1) 
else: 
	plt.ylim(0,0.08)
plt.xlim(xmin,xmax) 
plt.title("Shower Charge Distribution") 
plt.xlabel("Charge Deposited")
plt.ylabel("Fraction of Images")
plt.legend() 
plt.tight_layout()
if saveFig: 
	plt.savefig(histName) 
	print("Saved:", histName) 
plt.show()

