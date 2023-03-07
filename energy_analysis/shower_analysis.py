import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import numpy as np
import pywt 

## Print image to command line as ASCII  
np.set_printoptions(suppress=True) #print numpy as float 
def ASCII(shower): 
	for row in shower: 
		for num in row: 
			print(str(int(num)).zfill(2), end='') 
		print()

#fileName = "larcv_png_64_train_showers"
#fileName = "larcv_png_64_test_tracks"

fileName = "gen_epoch100a_v2_showers" 

#fileName = "Paul_Results_tracks"

showers = np.load("/home/zimani/GenNets/energy_analysis/npy_files/"+fileName+".npy")

## How many showers to iterate 
exitCount = -1

## Output Options 
saveSamples = False 
saveNPY = True
saveWave = False 
saveName = "thresh25_norm"
#saveName = "energy"

## Initilizations  
charges = [] 
wavelets = [] 


## Iterate All Showers 
for counter, shower in enumerate(showers): 
	
	charge = np.sum(shower) 
	
	if charge == 0: 
		continue 

	## Wavelet Lowest Compression Coefficients 
	wp = pywt.WaveletPacket2D(data=shower, wavelet='haar', mode='zero') 

	wavelet = [] 

	#wavelet.append(wp['aaaaaa'].data[0][0]) #[[val]]
	#wavelet.append(wp['hhhhhh'].data[0][0])
	#wavelet.append(wp['vvvvvv'].data[0][0])
	#wavelet.append(wp['dddddd'].data[0][0])

	#wavelet.append(wp['a'].data.sum()) 
	#wavelet.append(wp['h'].data.sum()) 
	#wavelet.append(wp['v'].data.sum()) 
	#wavelet.append(wp['d'].data.sum()) 
	
	if 0: 
		# Sum of Squares of Absolute Value = Energy 
		wavelet.append(np.sum(np.square(np.abs(wp['a'].data)))) 
		wavelet.append(np.sum(np.square(np.abs(wp['h'].data)))) 
		wavelet.append(np.sum(np.square(np.abs(wp['v'].data)))) 
		wavelet.append(np.sum(np.square(np.abs(wp['d'].data)))) 
			
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

	#coeffs = pywt.wavedec2(shower, 'db1', mode='zero') 
	#wavelet.append(coeffs[0][0][0]) 
	#for coeff in coeffs[1]: 
	#	wavelet.append(coeff[0][0]) 
	
	## Reformate Image to Pixel Location Data Centered at Origin 
	if counter == exitCount:    
		data = np.nonzero(shower)
		if len(data[0]) == 0: 
			print("Empty Track at #"+str(counter)) 
			continue
		#data = np.array([data[1], 64-data[0]]) #centered at 32,32
		data = np.array([data[1]-32, 32-data[0]])
		
		#new_wp = pywt.WaveletPacket2D(data=None, wavelet='db1', mode='zero')
		#coeffs2 = pywt.wavedec2(data, 'db1', mode='zero') 

		#coeffs2[0] = tuple([np.zeros_like(v) for v in coeffs2[0]])
		#coeffs2[1][0][0] = tuple([np.zeros_like(v) for v in coeffs2[1][0]][0])
		#coeffs2[1][1][0] = tuple([np.zeros_like(v) for v in coeffs2[1][1]][0])
		#coeffs2[1][2][0] == tuple([np.zeros_like(v) for v in coeffs2[1][2]][0])
		#print(coeffs2) 
		#print(coeffs2[0]) 
		#print(coeffs2[1][0][0]) 
		#print(coeffs2[1][1][0]) 
		#print(coeffs2[1][2][0]) 

		#rec = pywt.waverec2(coeffs2, 'db1', mode='zero') 
		

		#plt.scatter(data[0], data[1], label="Shower") 
		#plt.scatter(rec[0], rec[1], label='Reconstructed') 
		#plt.scatter(rec[0], rec[1], label='High Pass') 
		#plt.scatter(rec[0], rec[1], label='Low Pass 1') 
		#plt.scatter(rec[0], rec[1], label='Low Pass 2') 
		#plt.scatter(rec[0], rec[1], label='Low Pass 3') 
		#plt.legend()
		#plt.show() 

		#wp = pywt.WaveletPacket2D(data=shower, wavelet='db1', mode='zero')
		
		#print(wp['aaaaaa'].data) 


		#widths = np.arange(1,33) 
		#cwtmatr = signal.cwt(data.T, signal.ricker, widths) 
		#plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
        #   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())		
		#plt.show() 

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
			if saveWave: 
				plt.savefig("wavelet_images"+str(counter)+"_threshold.png")
			plt.show() 		

		exit() 
	
	## Save Image For Each Length 	
	if (saveSamples) and (charge not in charges):  
		im = Image.fromarray(track)
		im = im.convert("L")
		im.save("sample_charges/charge_"+str(charge)+"_sample_"+str(counter)+".png") 
		print("Saved Sample Charge "+str(charge))
	
	charges.append(charge) 
	wavelets.append(wavelet) 

## Histogram of Lengths 
if saveNPY: 
	np.save("hists/"+fileName+"_charges.npy", charges) 
	print("Saved: "+fileName+" Charges") 
	if saveWave: 
		np.save("hists/"+fileName+"_wavelets_"+saveName+".npy", wavelets) 
		print(" and wavelets") 

