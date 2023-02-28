import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

histPath = "/home/zimani/energy_analysis/hists/"

larcv = np.load(histPath+"larcv_png_64_train_tracks_angles.npy") 
larcv_len = np.load(histPath+"larcv_png_64_train_tracks_lengths.npy") 

gen = np.load(histPath+"ckpt_4_samples_grayscale_tracks_angles.npy") 
gen_len = np.load(histPath+"ckpt_4_samples_grayscale_tracks_lengths.npy") 

gen = np.degrees(gen) +180  
larcv = np.degrees(larcv)  + 180 

bins = 90 

plt.hist(larcv, bins=bins, density=True, histtype='step', label="LArTPC") 
plt.hist(gen, bins=bins, density=True, histtype='step', label="Generated") 
plt.legend(loc='lower right') 
plt.title("Track Angles") 
plt.ylabel("Fraction of Images")
plt.xlabel("Degrees")
#plt.xlim(-np.pi, np.pi)
plt.xlim(0, 360) 
plt.tight_layout()
plt.savefig("hist_angle_compare.png") 
plt.show() 

plt.clf() 

exit() 

## Circular Histogram 

Nl = larcv.size 
Ng = gen.size


binwidth = 4
all_bins = np.arange(0, 360, binwidth)
centers = 0.5*(all_bins[1:] + all_bins[:-1])  


l_count, _ = np.histogram(larcv, bins=all_bins) 
g_count, _ = np.histogram(gen, bins=all_bins) 

l_count_len, _ = np.histogram(larcv_len, bins=all_bins) 
g_count_len, _ = np.histogram(gen_len, bins=all_bins) 



## Circular Histogram 
ax = plt.subplot(111, polar=True) 
ax.bar(all_bins[:-1], l_count, width=binwidth, bottom=10, fill=False) 



#plt.savefig("hist_angle_compare.png") 

plt.show()


exit() 



## Plot Histograms 
plt.step(centers, l_count/Nl, linewidth=2, c="b", label="Training (N="+str(Nl)+")")
plt.step(centers, g_count/Ng, linewidth=2,  c="r", label="Generated (N="+str(Ng)+")") 

## Error Kewords Arguments 
errgs = dict(ls="none", 
			alpha=0.3,
			capsize=binwidth/2) 

## Plot Errors 
l_error = np.sqrt(l_count)/Nl
plt.errorbar(centers-binwidth/2, l_count/Nl, yerr=l_error, ecolor="b", **errgs)

g_error = np.sqrt(g_count)/Ng
plt.errorbar(centers-binwidth/2, g_count/Ng, yerr=g_error, ecolor="r", **errgs)

## Formatting 
plt.ylim(bottom=0) 
plt.xlim(0,xmax) 
plt.title("Track Length Distribution") 
plt.xlabel("Length of Track")
plt.ylabel("Fraction of Images")
plt.legend() 
plt.tight_layout()
#plt.savefig("hist_length_compare.png") 
#plt.show()

## Errors 
if 0: 
	SE =  np.sum( (l_count/Nl - g_count/Ng)**2)
	MSE = 1/np.size(all_bins) * SE 
	RMSE = np.sqrt(MSE) 
	chi = stats.chisquare(g_count[l_count != 0]/Ng, l_count[l_count != 0]/Nl)
	print("Squared Error =", str(round(SE,6)))
	print("Mean Squared Error =", str(round(MSE,6))) 
	print("Root Mean Squared Error =", str(round(RMSE,6)))
	print("Chi-squared (excluding zeros) =", str(round(chi[0],6)), "with p =", str(chi[1]))
	print() 


dof = 0 
chi_sqr = 0
RMSE = 0 
for lbin, gbin in zip(l_count/Nl, g_count/Ng):
	if (lbin + gbin) != 0: 
		dof += 1 
		chi_sqr += (gbin - lbin)**2 / (lbin + gbin) 
		RMSE += (gbin - lbin)**2 

print(chi_sqr)
print(dof) 
print(stats.chi2.sf(chi_sqr, dof)) 
print(np.sqrt(RMSE/dof))

