import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

##############
# Deprecated #
##############

histPath = "/home/zimani/energy_analysis/hists/"

base = np.load(histPath+"larcv_png_64_train_tracks_angles.npy") 
base = np.degrees(base)+180 
baseName = "LArTPC"

comp1 = np.load(histPath+"ckpt_4_samples_grayscale_tracks_angles.npy") 
comp1 = np.degrees(comp1)+180

comp2 = np.load(histPath+"ckpt_15_samples_grayscale_tracks_angles.npy") 
comp2 = np.degrees(comp2)+180


comps = [comp1, comp2] 
names = ['100 Epochs', '150 Epochs']
colors = ['red', 'green'] ##fix this

ymin = 0

binwidth = 6 
xmax = 360 
xmin = 0 
allBins = np.arange(-binwidth,xmax+binwidth,binwidth)
centers = 0.5*(allBins[1:] + allBins[:-1])  

saveFig = True 
histName = "hist_angle_compare.png"

## General Data Info
mins = [np.min(base)]
maxs = [np.max(base)]
for comp in comps:
    mins.append(np.min(comp))
    maxs.append(np.max(comp))
print("---")
print("Smallest Angle =  %d" % np.min(mins))
print("Most Charge = %d" % np.max(maxs))
print("Bins 0:%d | Bin Width = %d" % (xmax, binwidth))
print("---")

## Baseline Bin Data
countB, _ = np.histogram(base, bins=allBins)

nB = base.size
offB = nB - np.sum(countB)
perOffB = round(100 * offB/nB, 3)

print(baseName)
print("Images not shown =", str(offB), "("+str(perOffB)+"%)")
print("Degrees of Freedom (nonzero bins) =", str(np.count_nonzero(countB)))

plt.step(centers, countB/nB, linewidth=2, c="black", label=baseName+" (N="+str(nB)+")")

## Error Kewords Arguments
errgs = dict(ls="none",
            alpha=0.2,
            elinewidth=6.2, #Manually set 
            capsize=0)

## Plot Errors
plt.errorbar(centers-binwidth/2, countB/nB, yerr=np.sqrt(countB)/nB, ecolor="black", **errgs)


## Comparison Data
for i, comp in enumerate(comps):

    countC, _ = np.histogram(comp, bins=allBins)

    nC = comp.size
    offC = nC - np.sum(countC)
    perOffC = round(100 * offC/nC, 3)

    print()
    print(names[i])
    print("Images not shown =", str(offC), "("+str(perOffC)+"%)")

    # Plot Histogram
    plt.step(centers, countC/nC, linewidth=2, c=colors[i], label=names[i]+" (N="+str(nC)+")")

    # Error Bars
    plt.errorbar(centers-binwidth/2, countC/nC, yerr=np.sqrt(countC)/nC, ecolor=colors[i], **errgs)

    # Statistics
    ks2 = stats.ks_2samp(comp, base)
    print("Chi-square = %.5f" % ks2[0])
    print("   P = %.6g" % ks2[1])

    chi = stats.chisquare(countC[countB != 0] * (nB/nC), countB[countB != 0])
    print("Chi-square = %.5f" % chi[0])
    print("   P = %.6g" % chi[1])

print()


## Formatting 
plt.ylim(bottom=ymin) 
plt.xlim(xmin,xmax) 
plt.title("Track Angle Distribution") 
plt.xlabel("Degrees From X-Axis")
plt.ylabel("Fraction of Images")
plt.legend() 
plt.tight_layout()
if saveFig:
	plt.savefig(histName) 
plt.show()

