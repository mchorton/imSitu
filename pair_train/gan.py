# Training the image transformation logic using a neural network.
import json
import torch.nn as nn
import numpy as np
import torch
import random
import torch.utils.data as td
import torch.optim as optim
import torch.autograd as ag
import torch.nn.functional as F
import time
from ast import literal_eval as make_tuple
import utils.mylogger as logging
import pair_train.nn as pairnn

def makeGanDataTest():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData("data/models/nngandataTrain_test", "data/models/nngandataDev_test", pairnn.COMPFEATDIR, pairnn.VRNDATATEST, ganStyle=True)

def makeGanData():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData("data/models/nngandataTrain", "data/models/nngandataDev", pairnn.COMPFEATDIR, pairnn.VRNDATA, ganStyle=True)

class NetD(nn.Module):
  def __init__(self, inputSize, outputSize):
    super(NetD, self).__init__()
    self.input_layer = nn.Linear(inputSize, outputSize)
  def forward(self, x, y):
    """
    x - the image features
    y - the image labels, and the similarity score
    """
    # TODO 
    return self.input_layer(x)

class NetG(nn.Module):
  def __init__(self, inputSize, outputSize):
    super(NetG, self).__init__()
    self.input_layer = nn.Linear(inputSize, outputSize)
  def forward(self, x, y):
    """
    x - the image features
    y - the image labels, and the similarity score
    """
    # TODO 
    return self.input_layer(x)

def trainCGAN(datasetFileName = "data/models/nngandataTrain", ganFileName = "data/models/ganModel"):
  real_label = 1
  fake_label = 0
  logPer = 100
  epochs = 5
  batchSize = 128
  lr = 0.01
  beta1=0.9999

  dataset = torch.load(datasetFileName)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)
  
  nz = dataset[0][0].size(0)
  assert(nz == 1024), "Invalid dataset dimensions!"
  netD = NetD(nz, 1).cuda()
  netG = NetG(nz, nz).cuda()
  criterion = nn.BCELoss()
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda())
  label = ag.Variable(torch.FloatTensor(dataloader.batch_size).cuda())
  optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))

  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
      # Update D network
      netD.zero_grad()
      xdata, ydata = data

      xdata, ydata = ag.Variable(xdata.cuda()), ag.Variable(ydata.cuda())
      output = netD(xdata, ydata)
      label.data.fill_(real_label)
      errD_real = criterion(output, label[:len(output)])
      errD_real.backward()
      D_x = output.data.mean()

      noise.data.normal_(0, 1)
      fake = netG(noise, ydata)
      label.data.fill_(fake_label)
      output = netD(fake.detach(), ydata)
      errD_fake = criterion(output, label)
      errD_fake.backward()
      D_G_z1 = output.data.mean()
      errD = errD_real + errD_fake
      optimizerD.step()

      # Update G network
      netG.zero_grad()
      label.data.fill_(real_label)
      output = netD(fake, ydata)
      errG = criterion(output, label)
      errG.backward()
      D_G_z2 = output.data.mean()
      optimizerG.step()
      if i % logPer == logPer - 1:
        logging.getLogger(__name__).info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
  logging.getLogger(__name__).info("Saving (netD, netG) to %s" % ganFileName)
  torch.save((netD, netG), ganFileName)
