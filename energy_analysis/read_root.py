import os 
import ROOT 

path = "/home/zimani/particle_datasets/track_dataset/train_track.root"

rootFile = ROOT.TFile.Open(path,"READ")

tree = rootFile.Get("track")

#tree.Print()

#tree.Show(0)

data = tree.GetBranch("data")

data.Print()

#canvas = ROOT.TCanvas("canvas") 

#for key in histFile.GetListOfKeys():
#	print(key)
	#canvas.Clear() 
	#h = key.ReadObj()
	#h.Draw("h") 
	#canvas.SaveAs(rootName+"/"+h.GetName()+".png")
