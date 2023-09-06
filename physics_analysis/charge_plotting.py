import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


## TODO: add flags and streamline 


histPath = "/home/zimani/GenNets/energy_analysis/hists/"

base = np.load(histPath+"larcv_png_64_test_showers_charges.npy") 
baseName = "LArTPC Val" 

comp10 = np.load(histPath+"gen_epoch10_showers_charges.npy") 
#comp20 = np.load(histPath+"gen_epoch20_showers_charges.npy") 
#comp30 = np.load(histPath+"gen_epoch30_showers_charges.npy") 
#comp40 = np.load(histPath+"gen_epoch40_showers_charges.npy") 
comp50 = np.load(histPath+"gen_epoch50_showers_charges.npy") 
#comp100 = np.load(histPath+"gen_epoch100_showers_charges.npy") 
comp150 = np.load(histPath+"gen_epoch150_showers_charges.npy") 
#comp300 = np.load(histPath+"gen_epoch300_showers_charges.npy") 
vqvae = np.load(histPath+"VQVAE_showers_charges.npy") 

#comps = [comp20, comp30, comp40, comp50, comp100, comp150, comp300] 
#names = ['20 Epochs', '30 Epochs', '40 Epochs','50 Epochs', '100 Epochs', '150 Epochs', '300 Epochs']
#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange'] 

comps = [comp10, comp50, comp150]
styles = ['dashed', '-', 'dotted']
names = ['10 Epochs', '50 Epochs', '150 Epochs']
colors = ['g', 'r', 'b']

saveFig = True 
#histName = "hist_charge_compare_all_epochs.png" 
histName = "hist_charge_compare_key_epochs.png" 

saveStats = False # Save: [Name, K-S, P-Val, Chi-Square, P-Val]

showVQVAE = False
if showVQVAE: 
	histName = "hist_charge_compare_key_vqvae.png" 
	comps.append(vqvae) 
	names.append('VQ-VAE')
	styles.append('-') 
	colors.append('y') 

## Figure formatting  
plt.rcParams.update({'font.size': 20})
lWidth = 4
plt.figure(figsize=(12,8))

binwidth = 800
xmax = 37200
xmin = 0 
xmax = (binwidth * np.ceil(xmax/binwidth)) #round to binwidth multiple  
allBins = np.arange(-binwidth,xmax,binwidth)
centers = 0.5*(allBins[1:] + allBins[:-1])  

## General Data Info
mins = [np.min(base)]
maxs = [np.max(base)]
for comp in comps:
	mins.append(np.min(comp))
	maxs.append(np.max(comp))

## Baseline Bin Data
countB, _ = np.histogram(np.clip(base, allBins[0], allBins[-1]), bins=allBins)

# Number of Events

nB = base.size
offB = np.sum(np.logical_or([base<xmin], [xmax<base])) # Number events outside histogram range 
perOffB = round(100 * offB/nB, 3) # Percentage of events outside histogram range 

# Plot Histogram 
plt.step(centers, countB/nB, linewidth=lWidth, c="black", label=baseName+" (N="+str(nB)+")")

## Error Kewords Arguments
errgs = dict(ls="none",
	        alpha=0.3,
	        elinewidth=None,
	        capsize=2)

## Plot Errors
plt.errorbar(centers-binwidth/2, countB/nB, yerr=np.sqrt(countB)/nB, ecolor="black", **errgs)

print("Shower Charge")
print("Name, K-S, P-Val, Chi2, P-Val")

## Comparison Data
for i, comp in enumerate(comps):

	# Bin Data 
	countC, _ = np.histogram(np.clip(comp, allBins[0], allBins[-1]), bins=allBins)

	# Number of Events 
	nC = comp.size 
	offC = np.sum(np.logical_or([comp<xmin], [xmax<comp]))# Number events outside histogram range  
	perOffC = round(100 * offC/nC, 3) # Percentage of events outside histogram range 

	# Plot Histogram
	plt.step(centers, countC/nC, linewidth=lWidth, c=colors[i], linestyle=styles[i], label=names[i]+" (N="+str(nC)+")")

	# Error Bars
	plt.errorbar(centers-binwidth/2, countC/nC, yerr=np.sqrt(countC)/nC, ecolor=colors[i], **errgs)

	# Statistics
	ks2 = stats.ks_2samp(comp, base)
	chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
	print("%s,%.5f,%.6g,%.5f,%.6g" % (names[i], ks2[0], ks2[1], chi[0], chi[1]))

## Formatting 
#plt.ylim(bottom=0) 
if showVQVAE: 
	plt.ylim(0, 0.08) 
else: 
	plt.ylim(0,0.065)
plt.xlim(xmin,xmax) 
plt.title("Shower Charge Distribution") 
plt.xlabel("Charge Deposited")
plt.ylabel("Fraction of Images")
plt.xlim(0,xmax-binwidth*0.5)
plt.legend() 
plt.tight_layout()
if saveFig: 
	plt.savefig(histName) 
	print("Saved:", histName) 
plt.show()

