import matplotlib.pyplot as plt
from absl import flags 
from absl import app 
import numpy as np

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

	charges = [] 

	showers = np.load(FLAGS.filePath+FLAGS.fileName+".npy")

	# Iterate All Showers 
	for counter, shower in enumerate(showers): 
		
		charge = np.sum(shower) 
		charges.append(charge) 

	# Save Charges 
	if FLAGS.saveNPY: 
		np.save(outPath+fileName+"_charges.npy", charges) 
		print("Saved", fileName, "Charges") 

if __name__ == '__main__': 
	app.run(main)
