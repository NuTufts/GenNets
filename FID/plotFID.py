import matplotlib.pyplot as plt 
import numpy as np 

epochs = [10,20,30,40,50,60,100,150,300]

# Training Dataset (double_resnet[2])
tFID = [116.430020195818, 12.2718659439913, 3.90431489082357,
	3.06710681929949, 2.79309079926984, 2.74953105964787,
	2.73551396671267, 2.6542792638002, 2.766824749261]
tVQVAE = 42.0302090045512

# Validation Dataset (double_resnet[2]) 
vFID = [121.383403577486, 18.3063530684774, 10.2397028403388, 
	9.205781860038, 9.09464218995247, 8.87166402235636, 
	8.94381826410774, 8.70886043135631, 8.84371699914367]
vVQVAE = 48.14133925

# Train 10k Dataset 
tFID = [121.625770012309, 18.9580374502103, 10.2323459642991,
	9.1523467500183, 8.9416264096252, 8.99393980615258, 
	8.95135264451978,8.74600565156756,8.78559773570689]
vVQVAE10k = 47.54343442


plt.rcParams.update({'font.size': 16}) 

plt.figure(figsize=(10,6)) 
plt.plot(epochs, tFID, linestyle='-', marker='o', linewidth=3, markersize=10, label="Training Dataset (N=10k)") 
plt.plot(epochs, vFID, linestyle='--', marker='o', linewidth=3, markersize=10, label="Validation Dataset (N=10k)") 
plt.yscale('log') 
#plt.ylim(1, 15) 
plt.ylim(7, 150) 
plt.title("SSNet-FID") 
plt.xlabel("Epoch") 
plt.ylabel("Fréchet Inception Distance") 


#plt.xticks([0,10,20,30,40,50,60,100,150,200,250,300]) 
plt.xticks([10,30,50,100,150,200,250,300])

if 0: 
	# Annotate Epoch 20 
	#plt.annotate("{:.2f}".format(tFID[1]), (epochs[1], tFID[1]), textcoords="offset points", xytext=(-25,0), ha='center') 

	# Annotate Epoch 30 
	plt.annotate("{:.2f}".format(tFID[2]), (epochs[2], tFID[2]), textcoords="offset points", xytext=(25,0), ha='center') 
	#plt.annotate("{:.2f}".format(vFID[2]), (epochs[2], vFID[2]), textcoords="offset points", xytext=(30,0), ha='center') 

	# Annotate Epoch 40 
	plt.annotate("{:.2f}".format(tFID[3]), (epochs[3], tFID[3]), textcoords="offset points", xytext=(10,10), ha='center') 
	#plt.annotate("{:.2f}".format(vFID[3]), (epochs[3], vFID[3]), textcoords="offset points", xytext=(15,10), ha='center') 
	 
	# Annotate Epoch 50 
	plt.annotate("{:.2f}".format(tFID[4]), (epochs[4], tFID[4]), textcoords="offset points", xytext=(0,-20), ha='center') 
	#plt.annotate("{:.2f}".format(vFID[4]), (epochs[4], vFID[4]), textcoords="offset points", xytext=(0,-20), ha='center') 

	# Annotate 60, 100, 150, 300 
	for i in range(5, len(epochs)): 
		annoT = "{:.2f}".format(tFID[i]) 
		plt.annotate(annoT, (epochs[i], tFID[i]), textcoords="offset points", xytext=(0,10), ha='center') 
		annoV = "{:.2f}".format(vFID[i]) 
		#plt.annotate(annoV, (epochs[i], vFID[i]), textcoords="offset points", xytext=(0,10), ha='center') 

plt.legend() 
plt.tight_layout() 

plt.savefig("FID_plot.png") 

plt.show() 


