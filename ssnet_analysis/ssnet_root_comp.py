import matplotlib.pyplot as plt
from absl import flags 
from absl import app
import numpy as np
import ROOT 
import os 

set_inDir = "/home/zimani/GenNets/npy_files/"
set_outDir = "./key_epochs/"  
set_trainFile ="ssnet_hists_larcv_png_64_test"
set_trainName = "LArTPC Val"

FLAGS = flags.FLAGS 
flags.DEFINE_string('inDir', set_inDir, "Directory for input numpy files")
flags.DEFINE_string('outDir', set_outDir, "Directory for output histograms")
flags.DEFINE_bool('showVQVAE', False, "Option to include VQVAE")
flags.DEFINE_string('trainFile', set_trainFile, 'Training histogram to compare against')
flags.DEFINE_string('trainName', set_trainName, 'Name of training histogram to compare against')

def main(argv): 

	# Load ROOT files 
	trainFile = ROOT.TFile.Open(FLAGS.inDir+FLAGS.trainFile+".root", "READ") 
	epoch10 = ROOT.TFile.Open(FLAGS.inDir+"ssnet_hists_gen_epoch10.root", "READ")
	epoch50 = ROOT.TFile.Open(FLAGS.inDir+"ssnet_hists_gen_epoch50.root", "READ")
	epoch150 = ROOT.TFile.Open(FLAGS.inDir+"ssnet_hists_gen_epoch150.root", "READ")

	comps = [epoch10, epoch50, epoch150]  
	colors = [ROOT.kGreen, ROOT.kRed, ROOT.kBlue]
	names = ["10 Epochs", "50 Epochs", "150 Epochs"] 
	styles = [2, 1, 3]

	outDir = FLAGS.outDir 

	if FLAGS.showVQVAE: 
		vqvae = ROOT.TFile.Open(FLAGS.inDir+"ssnet_hists_VQVAE.root", "READ")
		outDir = "./VQVAE"
		comps.append(vqvae) 
		names.append("VQ-VAE")
		styles.append(10) 
		colors.append(ROOT.kOrange) 

	# Create output directory as needed 
	if not os.path.exists(outDir+"/"):
		os.makedirs(outDir+"/")

	nameList = ["hscore_abovethresh_track", "hscore_abovethresh_shower", 
				"hnpix_per_image_track", "hnpix_per_image_shower"]

	# Supress graphical output
	ROOT.gROOT.SetBatch(True)

	canvas = ROOT.TCanvas("canvas") 

	for histName in nameList: 

		canvas.Clear()	

		if "score" in histName: 
			canvas.SetLogy(True) 
		else: 
			canvas.SetLogy(False)  

		trainHist = trainFile.Get(histName)
		trainHist.Scale(10/trainHist.Integral(), "width") 
		trainHist.SetStats(0)
		trainHist.SetLineColor(ROOT.kBlack)
		trainHist.SetLineWidth(3) 
		trainHist.Draw("h") 
		
		if histName == "hnpix_per_image_track":
			#trainHist.GetYaxis().SetRangeUser(0,0.4)  
			trainHist.GetXaxis().SetRangeUser(0,200)  
		
		# Reformat Axis Titles - more like matplotlib 
		xTitle = trainHist.GetXaxis().GetTitle().title()
		yTitle = trainHist.GetYaxis().GetTitle().title()
		xTitle0 = xTitle.replace("Num", "Number of") 
		xTitle1 = xTitle0.replace("Scores", "Label Certainty") 
		yTitle1 = yTitle.replace("Of", "of") 
		trainHist.GetXaxis().SetTitle(xTitle1) 
		trainHist.GetYaxis().SetTitle(yTitle1) 
		trainHist.GetXaxis().CenterTitle(True)  
		trainHist.GetYaxis().CenterTitle(True)  
		
		# x1, y1, x2, y2 
		legend = ROOT.TLegend(0.65,0.65,0.85,0.85)
		legend.SetTextSize(0.04) 
		legend.AddEntry(trainHist, FLAGS.trainName) 
			
		# Comparison histograms 
		for i, comp in enumerate(comps): 
			compHist = comp.Get(histName)
			compHist.Scale(10/compHist.Integral(), "width")
			compHist.SetStats(0)
			compHist.SetLineColor(colors[i])
			compHist.SetLineStyle(styles[i])
			compHist.SetLineWidth(3)
			compHist.Draw("h, same") 
			legend.AddEntry(compHist, names[i]) 

		legend.Draw("same") 

		canvas.SaveAs(outDir+histName+"_comp.png")

if __name__ == '__main__': 
	app.run(main) 



