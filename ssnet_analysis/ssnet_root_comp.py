import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import root_numpy
import ROOT 
import os 

train ="ssnet_hists_larcv_png_64_test"
trainName = "LArTPC Val"

zpath = "/home/zimani/GenNets/npy_files/"

trainFile = ROOT.TFile.Open(zpath+train+".root", "READ") 

c10 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch10.root", "READ")

#c20 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch20.root", "READ")

#c30 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch30.root", "READ")

#c40 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch40.root", "READ")

c50 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch50.root", "READ")

#c100 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch100.root", "READ")

c150 = ROOT.TFile.Open(zpath+"ssnet_hists_gen_epoch150.root", "READ")

vqvae = ROOT.TFile.Open(zpath+"ssnet_hists_VQVAE.root", "READ")

comps = [c10, c50, c150]  

colors = [ROOT.kGreen, ROOT.kRed, ROOT.kBlue]
styles = [2, 1, 3]

names = ["10 Epochs", "50 Epochs", "150 Epochs"] 

outDir = "./key_epochs" 

showVQVAE = True
if showVQVAE: 
	outDir = "./VQVAE"
	comps.append(vqvae) 
	names.append("VQ-VAE")
	styles.append(10) 
	colors.append(ROOT.kOrange) 


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
	
	# Reformat Axis Titles 
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
	legend.AddEntry(trainHist, trainName) 
		
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

	canvas.SaveAs(outDir+"/"+histName+"_comp.png")

