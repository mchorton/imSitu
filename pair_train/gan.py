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
#from tqdm import tqdm
import tqdm
import sys
import os

def makeGanDataTest():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData("data/models/nngandataTrain_test", "data/models/nngandataDev_test", pairnn.COMPFEATDIR, pairnn.VRNDATATEST, ganStyle=True)

def makeGanData():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData("data/models/nngandataTrain", "data/models/nngandataDev", pairnn.COMPFEATDIR, pairnn.VRNDATA, ganStyle=True)

class NetD(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize):
    super(NetD, self).__init__()
    ydataLength = 6 * wESize + 6 * vrESize
    effectiveYSize = 20
    self.input_layer = nn.Linear(inputSize + effectiveYSize, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)

    self.ymap = nn.Linear(ydataLength, effectiveYSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)

    self.dropout = 0.

    logging.getLogger(__name__).info("Building Discriminator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (inputSize + ydataLength))
  def forward(self, x, y, train=False):
    """
    x - the image features
    y - the image labels, and the similarity score
    """
    context = y[:,:12] # Everything except the similarity score
    context_vectors = pairnn.getContextVectors(self.contextVREmbedding, self.contextWordEmbedding, context, len(x))
    mapped_cv = self.ymap(torch.cat(context_vectors, 1))
    x = torch.cat([x] + [mapped_cv], 1)

    x = F.dropout(F.leaky_relu(self.input_layer(x)), p=self.dropout, training=train)
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train)
      if i == 0: x_prev = x
      #add skip connections for depth
      if i > 0 and i % 2 == 0: 
        x = x + x_prev
        x_prev = x
    #x = F.dropout(F.relu(self.hidden2(x)), training=train)
    #x = F.dropout(F.relu(self.hidden3(x)), training=train)
    x = F.sigmoid(self.output(x))
    return x

class NetG(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize):
    super(NetG, self).__init__()
    ydataLength = 6 * wESize + 6 * vrESize
    effectiveYSize = 20
    self.inputSize = inputSize
    self.input_layer = nn.Linear(self.inputSize + effectiveYSize, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)

    self.ymap = nn.Linear(ydataLength, effectiveYSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)
    logging.getLogger(__name__).info("Building Generator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (self.inputSize + ydataLength))
  def forward(self, x, y, train=False):
    """
    x - the image noise
    y - the image labels, and the similarity score
    """
    assert(len(x) == len(y)), "invalid len(x)=%d, len(y)=%d" % (len(x), len(y))
    assert(len(x[0]) == self.inputSize), "invalid len(x[0])=%d, expect %d" % (len(x[0]), self.inputSize)
    assert(len(y[0]) == 13), "invalid len(y[0])=%d, expect 13" % (len(y[0]))
    context = y[:,:12] # Everything except the similarity score
    context_vectors = pairnn.getContextVectors(self.contextVREmbedding, self.contextWordEmbedding, context, len(x))
    catted = torch.cat(context_vectors, 1)
    mapped_cv = self.ymap(catted)
    x = torch.cat([x] + [mapped_cv], 1)

    x = F.dropout(F.leaky_relu(self.input_layer(x)), training=train) 
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train)
      if i == 0: x_prev = x
      #add skip connections for depth
      if i > 0 and i % 2 == 0: 
        x = x + x_prev
        x_prev = x
    #x = F.dropout(F.relu(self.hidden2(x)), training=train)
    #x = F.dropout(F.relu(self.hidden3(x)), training=train)
    x = self.output(x)
    return x

def trainCGANTest(datasetFileName = "data/models/nngandataTrain_test", ganFileName = "data/models/ganModel_test"):
  trainCGAN(datasetFileName, ganFileName)

def trainCGAN(datasetFileName = "data/models/nngandataTrain_gs_True", ganFileName = "data/models/ganModel", gpu_id=2, lr=1e-7, logPer=20, batchSize=128, epochs=5, saveDir=None, savePerEpoch=5):
  # Set up variables.
  real_label = 1
  fake_label = 0
  beta1=0.9999
  role2Int = torch.load("%s_role2Int" % datasetFileName)
  noun2Int = torch.load("%s_noun2Int" % datasetFileName)
  verb2Len = torch.load("%s_verb2Len" % datasetFileName)
  nVRs = max(role2Int.values()) + 1
  nWords = max(noun2Int.values()) + 1
  wESize = 128
  vrESize = 128
  hiddenSize = 4096
  depth = 2
  genDepth = 1
  nz = 100

  # Load variables, begin training.
  dataset = torch.load(datasetFileName)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)
  
  xSize = dataset[0][0].size(0)
  assert(xSize == 1024), "Invalid dataset dimensions!"

  netD = NetD(xSize, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  netG = NetG(nz, xSize, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  criterion = nn.BCELoss().cuda()
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  label = ag.Variable(torch.FloatTensor(dataloader.batch_size).cuda(gpu_id))
  optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))

  if saveDir is not None:
    if os.path.exists(saveDir):
      logging.getLogger(__name__).info("Save directory %s already exists; cowardly exiting..." % saveDir)
      sys.exit(1)
    os.makedirs(saveDir)

  logging.getLogger(__name__).info("Beginning training")
  for epoch in tqdm.tqdm(range(epochs), total=epochs, desc="Epochs completed: "):
    for i, data in tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Iterations completed: ", leave=False):
      # Update D network
      netD.zero_grad()
      xdata, ydata = data

      xdata, ydata = ag.Variable(xdata.cuda(gpu_id)), ag.Variable(ydata.cuda(gpu_id))
      output = netD(xdata, ydata, True) # TODO I think some of these calls for output should *NOT* use train=True
      label.data.fill_(real_label)
      errD_real = criterion(output.view(-1), label[:len(output)])  # TODO try adding the similarity score in here.
      errD_real.backward()
      D_x = output.data.mean()
      # D_x is how many of the images (all of which were real) were identified as real.

      noise.data.normal_(0, 1)
      fake = netG(noise[:len(ydata)], ydata, True)
      label.data.fill_(fake_label)
      output = netD(fake.detach(), ydata, True)
      errD_fake = criterion(output, label[:len(output)])
      errD_fake.backward()
      D_G_z1 = output.data.mean()
      # D_G_z1 is how many of the images (all of which were fake) were identified as real.
      errD = errD_real + errD_fake
      optimizerD.step()

      # Update G network
      netG.zero_grad()
      label.data.fill_(real_label)
      # TODO why is new output taken? Why do we use values from updated discriminator? Kind of makes sense...
      output = netD(fake, ydata, True) # TODO I called fake.detach()... ? What does that do?
      errG = criterion(output, label[:len(output)])
      errG.backward()
      D_G_z2 = output.data.mean()
      # D_G_z2 is how many of the images (all of which were fake) were identified as real, AFTER the discriminator update.
      optimizerG.step()
      if i % logPer == logPer - 1:
        tqdm.tqdm.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    # save the training files
    if saveDir is not None and epoch % savePerEpoch == (savePerEpoch - 1):
      checkpointName = "%s_epoch%d.pyt" % (os.path.basename(ganFileName), epoch)
      logging.getLogger(__name__).info("Saving checkpoint to %s" % checkpointName)
      torch.save((netD, netG), os.path.join(saveDir, checkpointName))
  logging.getLogger(__name__).info("Saving (netD, netG) to %s" % ganFileName)
  torch.save((netD, netG), ganFileName)

def evaluateGANModel(ganModelFile, datasetFileName):
  batchSize=512
  nz = 100
  """
  Evaluate a GAN model generator's image creation ability in a simple way.
  """
  netD, netG = torch.load(ganModelFile)
  netG = netG.cuda()
  netD = netD.cuda()
  dataset = torch.load(datasetFileName)
  # This dataset needs to be a regular-style dataset, not a gan-style one.
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)
  criterion = nn.MSELoss()
  runningTrue = 0.
  runningGimpy = 0.
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda())
  datasetSize = 0.
  for i, data in tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Iterations completed: "):
    expectedDataSize = 12 + 3 + 1024
    assert(len(data[0][0]) == expectedDataSize), "expected len(data[0][0])=%d to be %d" % (len(data[0][0]), expectedDataSize)
    expectedYDataSize = 1024 + 1
    assert(len(data[1][0]) == expectedYDataSize), "expected len(data[0][0])=%d to be %d" % (len(data[1][0]), expectedYDataSize)
    # 1. No labels on generator. Calculate square loss.
    xdata, featuresAndScore = data
    conditionalData = xdata[:,:12].cuda()
    scores = featuresAndScore[:,-1].cuda().contiguous().view(-1, 1)
    conditionalData = torch.cat([conditionalData, scores], 1)
    noise.data.normal_(0, 1)
    output = netG(noise[:len(conditionalData)], ag.Variable(conditionalData.cuda()))
    expectedFeatures = xdata[:,12 + 3:].cuda().contiguous()
    scores = scores.view(len(scores), 1).expand(output.size())

    correctLoss = criterion(
        ag.Variable(torch.mul(output.data, scores), requires_grad=False), 
        ag.Variable(torch.mul(expectedFeatures, scores), requires_grad=False))

    scores = featuresAndScore[:,-1].cuda().contiguous().view(-1, 1)
    conditionalData = torch.zeros([len(conditionalData), 12]).cuda()
    conditionalData = torch.cat([conditionalData, scores], 1)
    output = netG(noise[:len(conditionalData)], ag.Variable(conditionalData.cuda()))
    scores = scores.view(len(scores), 1).expand(output.size())
    gimpyLoss = criterion(
        ag.Variable(torch.mul(output.data, scores)), 
        ag.Variable(torch.mul(expectedFeatures, scores)))

    batchSize = len(conditionalData)
    datasetSize += batchSize

    runningTrue += correctLoss.data[0] * batchSize
    runningGimpy += gimpyLoss.data[0] * batchSize
  runningTrue /= datasetSize
  runningGimpy /= datasetSize
  logging.getLogger(__name__).info("True   Loss: %.4f" % runningTrue)
  logging.getLogger(__name__).info("zerodY Loss: %.4f" % runningGimpy)
