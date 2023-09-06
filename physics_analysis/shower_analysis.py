import matplotlib.pyplot as plt
from scipy import signal
from absl import flags 
from absl import app 
import numpy as np
import pywt #wavelet 

## Print event to command line as ASCII  
def ASCII(event): 
	np.set_printoptions(suppress=True) #Numpy formatting 
	for row in event: 
		for num in row: 
			print(str(int(num)).zfill(2), end='') 
		print()

set_fileName = "gen_epoch50_showers"
set_filePath = "/home/zimani/GenNets/npy_files/"
set_outPath = "./hists/"

FLAGS = flags.FLAGS 
flags.DEFINE_bool('saveNPY', True, 'Save shower charge as npy file')
flags.DEFINE_bool('saveWave', False, 'Save wavelet transform analysis (WiP)')
flags.DEFINE_string('fileName', set_fileName, 'Name of shower event file i.e. name_showers')
flags.DEFINE_string('filePath', set_filePath, 'Directory of shower events')
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
	charges = [] 
	wavelets = [] 

	showers = np.load(FLAGS.filePath+FLAGS.fileName+".npy")

	## Iterate All Showers 
	for counter, shower in enumerate(showers): 
		
		charge = np.sum(shower) 
		charges.append(charge) 

		if charge == 0: 
			continue
		
		## Wavelet Analysis (WiP)
		if FLAGS.saveWave: 

			## Reformate Image to Pixel Location Data Centered at Origin 
			if 0:
				data = np.nonzero(shower)
				if len(data[0]) == 0: 
					print("Empty Track at #"+str(counter)) 
					continue
				#data = np.array([data[1], 64-data[0]]) #centered at 32,32
				data = np.array([data[1]-32, 32-data[0]])

			## Wavelet Lowest Compression Coefficients 
			wp = pywt.WaveletPacket2D(data=shower, wavelet='haar', mode='zero') 

			# Set wavelet name for saving 
			waveName = "thresh25_norm"

			wavelet = [] 
			threshold = 25 
			if 1: 
				dataA = np.clip(abs(wp['a'].data), 0, 255)  
				dataH = np.clip(abs(wp['h'].data), 0, 255)  
				dataV = np.clip(abs(wp['v'].data), 0, 255)  
				dataD = np.clip(abs(wp['d'].data), 0, 255)  

				dataA[dataA < threshold] = 0 
				dataH[dataH < threshold] = 0 
				dataV[dataV < threshold] = 0 
				dataD[dataD < threshold] = 0 
				
				pixelNorm = np.count_nonzero(shower) 
				wavelet.append(np.count_nonzero(dataA)/pixelNorm) 
				wavelet.append(np.count_nonzero(dataH)/pixelNorm)
				wavelet.append(np.count_nonzero(dataV)/pixelNorm)
				wavelet.append(np.count_nonzero(dataD)/pixelNorm) 
			
				if 1: 
					#wp = pywt.WaveletPacket2D(data=shower, wavelet='haar', mode='zero') 
					
					strA = '' 
					strH = '' 
					strV = ''
					strD = '' 

					fig = plt.figure(figsize=(15, 9)) 
					for i in range(wp.maxlevel+1): 
						axA = fig.add_subplot(4, 7, 1+i)
						axH = fig.add_subplot(4, 7, 8+i)
						axV = fig.add_subplot(4, 7, 15+i)
						axD = fig.add_subplot(4, 7, 22+i)
						
			
						dataA = np.clip(abs(wp[strA].data), 0, 255)  
						dataH = np.clip(abs(wp[strH].data), 0, 255)  
						dataV = np.clip(abs(wp[strV].data), 0, 255)  
						dataD = np.clip(abs(wp[strD].data), 0, 255)  

						threshold = 50
						dataA[dataA < threshold] = 0 
						dataH[dataH < threshold] = 0 
						dataV[dataV < threshold] = 0 
						dataD[dataD < threshold] = 0 

						axA.imshow(dataA, vmin=0, vmax=255)  
						axH.imshow(dataH, vmin=0, vmax=255)  
						axV.imshow(dataV, vmin=0, vmax=255)  
						axD.imshow(dataD, vmin=0, vmax=255)  

						axA.set_xticks([]) 
						axA.set_yticks([]) 
						axH.set_xticks([]) 
						axH.set_yticks([]) 
						axV.set_xticks([]) 
						axV.set_yticks([]) 
						axD.set_xticks([]) 
						axD.set_yticks([]) 

						axA.set_title("Depth "+str(i))			

						# Sum 
						#axA.set_xlabel(round(wp[strA].data.sum(),5))
						#axH.set_xlabel(round(wp[strH].data.sum(),5))
						#axV.set_xlabel(round(wp[strV].data.sum(),5))
						#axD.set_xlabel(round(wp[strD].data.sum(),5))
						
						# Abs Average 
						#axA.set_xlabel(round(np.abs(wp[strA].data).mean(),5))
						#axH.set_xlabel(round(np.abs(wp[strH].data).mean(),5))
						#axV.set_xlabel(round(np.abs(wp[strV].data).mean(),5))
						#axD.set_xlabel(round(np.abs(wp[strD].data).mean(),5))

						axA.set_xlabel(np.count_nonzero(dataA)) 
						axH.set_xlabel(np.count_nonzero(dataH)) 
						axV.set_xlabel(np.count_nonzero(dataV)) 
						axD.set_xlabel(np.count_nonzero(dataD)) 

						strA += 'a' 
						strH += 'h'
						strV += 'v'
						strD += 'd' 
						
						if i == 0: 
							axA.set_ylabel("Approximate") 
							axH.set_ylabel("Horizontal")
							axV.set_ylabel("Vertical")
							axD.set_ylabel("Diagonal")
					
					#axA.text(-0.2, 0, round(wp[strA[:-1]].data[0][0],5), c='white')
					#axH.text(-0.2, 0, round(wp[strH[:-1]].data[0][0],5), c='white')
					#axV.text(-0.2, 0, round(wp[strV[:-1]].data[0][0],5), c='white')
					#axD.text(-0.2, 0, round(wp[strD[:-1]].data[0][0],5), c='white')

					fig.tight_layout() 
					if 0: 
						plt.savefig("wavelet_images"+str(counter)+"_threshold.png")
					plt.show()
					exit() 
		
				wavelets.append(wavelet) 

	## Histogram of Lengths 
	if FLAGS.saveNPY: 
		np.save(outPath+fileName+"_charges.npy", charges) 
		print("Saved", fileName, "Charges") 
		if FLAGS.saveWave: 
			np.save(outPath+fileName+"_wavelets_"+waveName+".npy", wavelets) 
			print("Saved Wavelets") 


if __name__ == '__main__': 
	app.run(main)
