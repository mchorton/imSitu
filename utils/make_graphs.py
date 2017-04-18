#!/usr/bin/python
# Script to make the GAN and NN training graphs.
"""
GANFILE = "sets/proj/data/filtered_gandata.csv"
NNFILE = "sets/proj/data/filtered_nndata.csv"

GANOUT = "sets/proj/data/gandata.eps"
NNOUT = "sets/proj/data/nndata.eps"
NNOUT_PAPER = "sets/proj/data/nndata_paper.eps"
"""
import os
import pandas

def makeGraphs():
  makeNNGraph()
  makeGANGraph()

def makeNNGraph():
  import pandas
  import cs517.lib.plotutils as pu
  data = pandas.read_csv(NNFILE)
  print "data is \n%s" % str(data)

  pu.makeTwoPlot(data["Epoch"], data["Train_Loss"], data["Dev_Loss"], "Train Loss", "Dev Loss", "NN Loss", "Epoch", "Loss", "linear", "linear", NNOUT, 40, 40, 2, 5)
  pu.makeTwoPlot(data["Epoch"], data["Train_Loss"], data["Dev_Loss"], "Train Loss", "Dev Loss", "NN Loss", "Epoch", "Loss", "linear", "linear", NNOUT_PAPER, 25, 25, 2, 5)

def makeGANGraph(infile):
  inname, inext = os.path.splitext(infile)
  outfile = inname + ".eps"
  other_outfile = inname + ".jpg"
  import pandas
  import plotutils as pu
  data = pandas.read_csv(infile)

  pu.makeTwoPlot(data["Epoch"], data["D_Loss"], data["G_Loss"], "D Loss", "G Loss", "CGAN Loss", "Epoch", "Loss", "linear", "linear", outfile, 25, 25, 2, 5)
  pu.makeTwoPlot(data["Epoch"], data["D_Loss"], data["G_Loss"], "D Loss", "G Loss", "CGAN Loss", "Epoch", "Loss", "linear", "linear", other_outfile, 25, 25, 2, 5)



def makeFullGANGraph(infile):
  inname, inext = os.path.splitext(infile)
  outfile = inname + ".eps"
  other_outfile = inname + ".jpg"
  import pandas
  import plotutils as pu
  data = pandas.read_csv(infile)

  if len(data["Epoch"] > 0):
    pu.makeNPlot(data["Epoch"], [data["D_Loss"], data["G_Loss"], data["D(x)"], data["D(G(z))_1"]], ["D Loss", "G Loss", "D(x)", "D(G(z))"], "CGAN Loss", "Epoch", "Loss", "linear", "linear", outfile, 10, 10, 1, 1)
    pu.makeNPlot(data["Epoch"], [data["D_Loss"], data["G_Loss"], data["D(x)"], data["D(G(z))_1"]], ["D Loss", "G Loss", "D(x)", "D(G(z))"], "CGAN Loss", "Epoch", "Loss", "linear", "linear", other_outfile, 10, 10, 1, 1)

if __name__ == '__main__':
  import sys
  makeFullGANGraph(sys.argv[1])
