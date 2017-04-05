# Training the image transformation logic using a neural network.
import json
import scipy.stats as ss
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
import collections
import copy
import constants as pc # short for pair_train constants

def makeGanDataTest():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData(
      "data/models/nngandataTrain_test", "data/models/nngandataDev_test",
      pairnn.COMPFEATDIR, pairnn.VRNDATATEST, style="gan")

def makeGanData():
  logging.getLogger(__name__).info("Making GAN data")
  pairnn.makeData("data/models/nngandataTrain", "data/models/nngandataDev",
      pairnn.COMPFEATDIR, pairnn.VRNDATA, style="gan")

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
  #def handleForward(*args, **kwargs):
  #  return self.forward(args, kwargs)

class NetG(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize, dropout=0.):
    super(NetG, self).__init__()
    ydataLength = 6 * wESize + 6 * vrESize
    effectiveYSize = 20
    self.inputSize = inputSize
    self.input_layer = nn.Linear(self.inputSize + effectiveYSize, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)
    self.dropout = dropout

    self.ymap = nn.Linear(ydataLength, effectiveYSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)
    logging.getLogger(__name__).info("Building Generator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (self.inputSize + ydataLength))
  def forward(self, x, y, train=False, ignoreCond=False):
    """
    x - the image noise
    y - the image labels, and the similarity score
    """
    ignoreNoise = True if self.inputSize < 1 else False
    assert(len(x) == len(y)), "invalid len(x)=%d, len(y)=%d" % (len(x), len(y))
    if not ignoreNoise:
      assert(len(x[0]) == self.inputSize), "invalid len(x[0])=%d, expect %d" % (len(x[0]), self.inputSize)
    assert(len(y[0]) == 13 or len(y[0]) == 12), "invalid len(y[0])=%d, expect 12 or 13" % (len(y[0]))
    context = y[:,:12] # Everything except the similarity score
    if ignoreCond:
      context = context.clone()
      context.data.fill_(0)
    context_vectors = pairnn.getContextVectors(self.contextVREmbedding, self.contextWordEmbedding, context, len(x))
    catted = torch.cat(context_vectors, 1)
    toCat = [self.ymap(catted)]
    if not ignoreNoise:
      toCat = [x] + toCat
    x = torch.cat(toCat, 1)

    x = F.dropout(F.leaky_relu(self.input_layer(x)), training=train, p=self.dropout) 
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train, p=self.dropout)
      if i == 0: x_prev = x
      #add skip connections for depth
      if i > 0 and i % 2 == 0: 
        x = x + x_prev
        x_prev = x
    x = self.output(x)
    return x
#def handleForward(*args, **kwargs):
#  return self.forward(args, kwargs)

def checkDatasetStyle(xdata, ydata):
  if len(xdata[0]) == (12 + 3 + pc.IMFEATS) and len(ydata[0]) == (pc.IMFEATS + 1):
    return ""
  elif len(xdata[0]) == (pc.IMFEATS) and len(ydata[0]) == (12 + 1):
    return "gan"
  raise ValueError("Invalid data set with xdata[0] dim=%d, ydata[0] dim=%d" % (len(xdata[0]), len(ydata[0])))

class TrGanD(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize):
    super(TrGanD, self).__init__()
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
    # Note: these arguments are deliberately reversed from NetG. Also, 'x'
    # has the scores, which is another difference. This is dictated by the
    # data format in pair_train/nn.py, for "style == 'trgan'"
    x - the image features, and the scores (unused)
    y - the image labels, the transformed role, the first noun, the second noun, and the input image features
    """
    x = x[:,:pc.IMFEATS] # get rid of the scores.
    assert(len(x) == len(y))
    assert(len(x[0]) == pc.IMFEATS), "Invalid len(x[0])=%d" % len(x[0])
    assert(len(y[0]) == 12 + 3 + pc.IMFEATS)

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
#  def handleForward(self, xdata, ydata, *args, **kwargs):
#    """
#    xdata - an x sample, as from "trgan" style gan input
#    ydata - a y sample, as from "trgan" style gan input
#    """
#    # Omit the score from xdata
#    self.forward(ydata, xdata[:,:12], *args, **kwargs)

class TrGanG(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize, dropout=0.):
    super(TrGanG, self).__init__()
    self.ydataLength = (6 * wESize + 6 * vrESize) + (vrESize) + (2 * wESize) + pc.IMFEATS
    effectiveYSize = self.ydataLength # TODO or shall I just concat with the noise?
    self.inputSize = inputSize
    self.input_layer = nn.Linear(self.inputSize + effectiveYSize, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)
    self.dropout = dropout

    # TODO add power here?
    #self.ymap = nn.Linear(self.ydataLength, effectiveYSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)
    logging.getLogger(__name__).info("Building Generator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (self.inputSize + self.ydataLength))
    # TODO should I use padding_idx on the embeddings, to make sure that removing the conditional
    # corresponds to the correct representation?
  def forward(self, z, y, train=False, ignoreCond=False):
    """
    z - the image noise
    y - the image labels, the transformed role, the first noun, the second noun,
    and the input image features
    """
    # TODO @mchorton I ignored att_w, att_r, att_b from Mark's NN code.
    ignoreNoise = True if self.inputSize < 1 else False
    if not ignoreNoise:
      assert(len(z[0]) == self.inputSize), \
          "invalid len(z[0])=%d, expect %d" % (len(z[0]), self.inputSize)
    assert(len(y[0]) == (12 + 3 + pc.IMFEATS)), \
        "invalid len(y[0])=%d, expect %d" % (len(y[0]), 12 + 3 + pc.IMFEATS)
    assert(len(z) == len(y)), "invalid len(z)=%d, len(y)=%d" % (len(z), len(y))

    batchsize = len(z)
    annotations = pairnn.getContextVectors(
        self.contextVREmbedding, self.contextWordEmbedding, y[:,:12], batchsize)
    role = self.contextVREmbedding(y[:,12].long()).view(batchsize, -1)
    n1 = self.contextWordEmbedding(y[:,13].long()).view(batchsize, -1)
    n2 = self.contextWordEmbedding(y[:,14].long()).view(batchsize, -1)
    img = y[:,15:15 + pc.IMFEATS]

    context = torch.cat(annotations + [role, n1, n2, img], 1)
    if ignoreCond:
      context = context.clone()
      context.data.fill_(0)

    assert(context.size(1) == self.ydataLength), \
        "context size %d, expected %d" % (context.size(1), self.ydataLength)
    toCat = [context]
    if not ignoreNoise:
      toCat = [z] + toCat
    x = torch.cat(toCat, 1)

    x = F.dropout(
        F.leaky_relu(self.input_layer(x)), training=train, p=self.dropout) 
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train, p=self.dropout)
      if i == 0: x_prev = x
      #add skip connections for depth
      if i > 0 and i % 2 == 0: 
        x = x + x_prev
        x_prev = x
    x = self.output(x)
    return x

def trainCGANTest(datasetFileName = "data/models/nngandataTrain_test", ganFileName = "data/models/ganModel_test"):
  trainCGAN(datasetFileName, ganFileName)

def trainCGAN(
    datasetFileName = "data/models/nngandataTrain_gs_True",
    ganFileName = "data/models/ganModel", gpu_id=2, lr=1e-7, logPer=20,
    batchSize=128, epochs=5, saveDir=None, savePerEpoch=5, lam=0.5, nz=100,
    hiddenSize=pc.IMFEATS, style="gan", **kwargs):
  # Set up variables.
  beta1=0.9999
  role2Int = torch.load("%s_role2Int" % datasetFileName)
  noun2Int = torch.load("%s_noun2Int" % datasetFileName)
  verb2Len = torch.load("%s_verb2Len" % datasetFileName)
  nVRs = max(role2Int.values()) + 1
  nWords = max(noun2Int.values()) + 1
  wESize = 128
  vrESize = 128

  gdropout = kwargs.get("gdropout", 0.)
  depth = kwargs.get("depth", 2)
  genDepth = kwargs.get("genDepth", 2)
  decayPer = kwargs.get("decayPer", 10)
  decayRate = kwargs.get("decayRate", 0.5)
  dUpdates = kwargs.get("dUpdates", 1)
  gUpdates = kwargs.get("gUpdates", 1)
  # Whether to ignore the conditioning variable. Useful for baselines.
  ignoreCond = kwargs.get("ignoreCond", False) 

  # Load variables, begin training.
  dataset = torch.load(datasetFileName)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=4)
  
  xSize = pc.IMFEATS # Size of generated images.
  #assert(xSize == pc.IMFEATS), "Invalid dataset dimensions!"

  # TODO this should depend a bit on how the data looks.
  # TODO double check parzen window code. Why bad?
  if style == "gan":
    netG = NetG(nz, xSize, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize, gdropout).cuda(gpu_id)
    netD = NetD(xSize, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  elif style == "trgan":
    netG = TrGanG(nz, xSize, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize, gdropout).cuda(gpu_id)
    netD = TrGanD(xSize, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  criterion = nn.BCELoss().cuda(gpu_id)
  l1Criterion = nn.L1Loss().cuda(gpu_id)
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  label = ag.Variable(torch.FloatTensor(dataloader.batch_size).cuda(gpu_id))
  optimizerD = optim.SGD(netD.parameters(), lr=lr)
  optimizerG = optim.SGD(netG.parameters(), lr=lr)

  if saveDir is not None:
    if os.path.exists(saveDir):
      logging.getLogger(__name__).info("Save directory %s already exists; cowardly exiting..." % saveDir)
      sys.exit(1)
    os.makedirs(saveDir)



  for epoch in tqdm.tqdm(range(epochs), total=epochs, desc="Epochs completed: "):
    for i, data in tqdm.tqdm(
        enumerate(dataloader, 0), total=len(dataloader), desc="Iterations completed: ", leave=False):
      # TODO change this to depend on the network.
      # Or, push all complexity into the network class itself?
      # Or, make a trainer that has a nn.Module as a member? Nah.
      networkUpdateStep(netD, netG, data, dUpdates, gUpdates, noise, label, gpu_id, ignoreCond, i, style, optimizerD, optimizerG, criterion, l1Criterion, logPer, lam, epoch, epochs, len(dataloader))
        
    # save the training files
    # TODO make this less greedy so saving doesn't cause crashes.
    if saveDir is not None and epoch % savePerEpoch == (savePerEpoch - 1):
      checkpointName = "%s_epoch%d.pyt" % (os.path.basename(ganFileName), epoch)
      logging.getLogger(__name__).info("Saving checkpoint to %s" % checkpointName)
      torch.save((netD, netG), os.path.join(saveDir, checkpointName))

    if epoch % decayPer == decayPer - 1:
      for param_group in optimizerG.param_groups:
        logging.getLogger(__name__).info("optG: prev rate: %.4f" % param_group['lr'])
        param_group['lr'] = param_group['lr'] * decayRate
        logging.getLogger(__name__).info("optG: new  rate: %.4f" % param_group['lr'])
      for param_group in optimizerD.param_groups:
        logging.getLogger(__name__).info("optD: prev rate: %.4f" % param_group['lr'])
        param_group['lr'] = param_group['lr'] * decayRate
        logging.getLogger(__name__).info("optD: new  rate: %.4f" % param_group['lr'])
  logging.getLogger(__name__).info("Saving (netD, netG) to %s" % ganFileName)
  torch.save((netD, netG), ganFileName)

def getScores(ydata, style):
  if style == "gan":
    scores = ydata[:,12] #DIFF
  elif style == "trgan":
    scores = ydata[:,pc.IMFEATS]
  else:
    raise ValueError("Invalid style '%s'" % style)
  return scores

# TODO
# what is xdata in all different cases?
# what should TrGanD/G take in?
# What should in/out be?
def networkUpdateStep(
    netD, netG, data, dUpdates, gUpdates, noise, label, gpu_id, ignoreCond, i, 
    style, optimizerD, optimizerG, criterion, l1Criterion, logPer, lam, epoch,
    epochs, nSamples):
  real_label = 1
  fake_label = 0
  # Update D network
  for _ in range(dUpdates):
    netD.zero_grad()
    xdata, ydata = data
    xdata, ydata = ag.Variable(xdata.cuda(gpu_id)), \
        ag.Variable(ydata.cuda(gpu_id))
    if style == "trgan":
      temp = xdata
      xdata = ydata
      ydata = temp

    # get scores from ydata
    scores = getScores(ydata, style)
    output = netD(xdata, ydata, True)
    label.data.fill_(real_label)
    errD_real = criterion(
        torch.mul(output.view(-1), scores),
        torch.mul(label[:len(output)], scores))
    errD_real.backward()
    # D_x is how many of the images (all of which were real) were identified as
    # real.
    D_x = output.data.mean()

    noise.data.normal_(0, 1)
    fake = netG(noise[:len(ydata)], ydata, True, ignoreCond=ignoreCond)
    label.data.fill_(fake_label)
    output = netD(fake, ydata, True)
    errD_fake = criterion(
        torch.mul(output, scores),
        torch.mul(label[:len(output)], scores))
    errD_fake.backward()
    # D_G_z1 is how many of the images (all of which were fake) were identified 
    # as real.
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step()

  for _ in range(gUpdates):
    # Update G network
    noise.data.normal_(0, 1)
    netG.zero_grad()
    fake = netG(noise[:len(ydata)], ydata, True, ignoreCond=ignoreCond)
    label.data.fill_(real_label)
    output = netD(fake, ydata, True)
    errG = criterion(
        torch.mul(output, scores),
        torch.mul(label[:len(output)], scores))
    errG.backward()
    D_G_z2 = output.data.mean()
    # D_G_z2 is how many of the images (all of which were fake) were identified
    # as real, AFTER the discriminator update.

    fake = netG(noise[:len(ydata)], ydata, True, ignoreCond=ignoreCond)
    resizescores = scores.contiguous().view(-1, 1).expand(fake.size())
    # because of the swapping, ydata[:,:1024] is the image we hoped to produce
    if style == "trgan":
      trueImg = xdata[:,:1024]
    elif style == "gan":
      trueImg = xdata
    else:
      raise ValueError("Invalid style %s" % style)
    errG_L1 = l1Criterion(
        torch.mul(torch.mul(fake, lam), resizescores),
        torch.mul(torch.mul(trueImg, lam), resizescores))
    errG_L1.backward()
    optimizerG.step()

  if i % logPer == logPer - 1:
    tqdm.tqdm.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, nSamples,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    logging.getLogger(__name__).info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, nSamples,
                 errD.data[0], errG.data[0], errG_L1.data[0], D_x, D_G_z1, D_G_z2))

def evaluateGANModel(ganModelFile, datasetFileName, gpu_id=2):
  batchSize=16
  """
  Evaluate a GAN model generator's image creation ability in a simple way.
  """
  netD, netG = torch.load(ganModelFile)

  nz = netG.inputSize

  netG = netG.cuda(gpu_id)
  netD = netD.cuda(gpu_id)
  dataset = torch.load(datasetFileName)
  # This dataset needs to be a regular-style dataset, not a gan-style one.
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batchSize, shuffle=True, num_workers=4)
  criterion = nn.MSELoss()
  runningTrue = 0.
  runningGimpy = 0.
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  datasetSize = 0.
  for i, data in tqdm.tqdm(
        enumerate(dataloader, 0), total=len(dataloader),
        desc="Iterations completed: "):
    expectedDataSize = 12 + 3 + pc.IMFEATS
    assert(len(data[0][0]) == expectedDataSize), \
        "expected len(data[0][0])=%d to be %d" % \
        (len(data[0][0]), expectedDataSize)
    expectedYDataSize = pc.IMFEATS + 1
    assert(len(data[1][0]) == expectedYDataSize), \
        "expected len(data[0][0])=%d to be %d" % \
        (len(data[1][0]), expectedYDataSize)
    # 1. No labels on generator. Calculate square loss.
    xdata, featuresAndScore = data
    conditionalData = xdata[:,:12].cuda(gpu_id)
    scores = featuresAndScore[:,-1].cuda(gpu_id).contiguous().view(-1, 1)
    conditionalData = torch.cat([conditionalData, scores], 1)
    noise.data.normal_(0, 1)
    output = netG(
        noise[:len(conditionalData)], ag.Variable(conditionalData.cuda(gpu_id),
            volatile=True), False, ignoreCond=False)
    expectedFeatures = xdata[:,12 + 3:].cuda(gpu_id).contiguous()
    scores = scores.view(len(scores), 1).expand(output.size())

    correctLoss = criterion(
        ag.Variable(torch.mul(output.data, scores), requires_grad=False), 
        ag.Variable(torch.mul(expectedFeatures, scores), requires_grad=False))

    batchSize = len(conditionalData)
    datasetSize += batchSize

    runningTrue += correctLoss.data[0] * batchSize
  runningTrue /= datasetSize
  logging.getLogger(__name__).info("True   Loss: %.4f" % runningTrue)

class GanDataLoader(object):
  def __init__(self, filename):
    self.filename = filename
    self.data = torch.load(self.filename)
    self.validate()
  def validate(self):
    if len(self.data) == 0:
      raise Exception("GanDataLoader", "No data found")
    x, y = self.data[0]
    assert(len(x) == pc.IMFEATS), "Invalid 'x' dimension '%d'" % len(x)
    assert(len(y) == 13), "Invalid 'y' dimension '%d'" % len(y)

def getGANChimeras(netG, nTestPoints, yval, gpu_id=0, dropoutOn=True):
  batchSize = 64
  out = []
  nz = netG.inputSize
  noise = ag.Variable(
      torch.FloatTensor(64, nz), requires_grad=False).cuda(gpu_id)
  yvar = ag.Variable(yval.expand(batchSize, yval.size(1))).cuda(gpu_id)
  for i in range(nTestPoints):
    noise.data.normal_(0, 1)
    if i * batchSize > nTestPoints:
      break
    out = torch.cat(out + [netG(noise, yvar, dropoutOn, ignoreCond=False)], 0)
    out = [out]
  return out[0]

# TODO this should basically work, but has a lot of print statements.
def parzenWindowCheck(ganModelFile, ganDevSet, **kwargs):
  """
  TODO: I need to find out why the values are always nan or 0.
  1. Try more samples in the PDF? might not help...
  2. Check if 'y' values are unique; check multiplicity.
  3. Or, maybe the model is just bad...
  """
  # Set up some values.
  gpu_id = kwargs.get("gpu_id", 2)
  logPer = kwargs.get("logPer", 1e12)
  bw_method = kwargs.get("bw_method", "scott")
  _, netG = torch.load(ganModelFile)
  netG = netG.cuda(gpu_id)
  gDiskDevLoader = GanDataLoader(ganDevSet)
  nSamples = kwargs.get("nSamples", 100)
  nTestSamples = kwargs.get("nTestSamples", 100)
  batchSize = kwargs.get("batchSize", 32)
  nz = netG.inputSize


  # map from conditioned value to a list of datapoints that match that value.
  yValsToXpoints = collections.defaultdict(list)
  ymap = collections.defaultdict(list)

  # This batch_size must be one, because I don't iterate over ydata again.
  devDataLoader = torch.utils.data.DataLoader(gDiskDevLoader.data, batch_size=1, shuffle=False, num_workers=0) # TODO
  logging.getLogger(__name__).info("Testing dev data against parzen window fit")
  for i, data in tqdm.tqdm(enumerate(devDataLoader, 0), total=len(devDataLoader), desc="Iterations completed: "):
    if i >= nTestSamples:
      break
    xdata, ydata = data
    # Herein lies the problem TODO
    ydata = ydata[:,:12] 
    #logging.getLogger(__name__).info("Y size")
    #logging.getLogger(__name__).info(ydata.size())

    #ydata = 
    #ydata = ydata[:1] # ignore score
    # xdata is 1xpc.IMFEATS
    kde_input = torch.transpose(xdata, 0, 1).numpy()
    # kde_input is pc.IMFEATSx1
    if ydata in yValsToXpoints:
      logging.getLogger(__name__).info("HAD POINT %s" % str(ydata))
    yValsToXpoints[ydata].append(kde_input)
    if i <= 0:
      logging.getLogger(__name__).info("x=%s" % str(xdata))
      logging.getLogger(__name__).info("y=%s" % str(ydata))

    key = tuple(*ydata.numpy().tolist())
    if key in ymap:
      logging.getLogger(__name__).info("i=%d, found repeat key %s" % (i, str(key)))
    ymap[key].append(kde_input)





  logging.getLogger(__name__).info(len(yValsToXpoints))
  agg = collections.defaultdict(int)
  # Maybe I shouldn't call len()?
  for k, v in yValsToXpoints.iteritems():
    #logging.getLogger(__name__).info("k=%s, v=%s" % (str(k), str(v)))
    agg[len(v)] += 1
  logging.getLogger(__name__).info(agg)

  probs = {}
  for i, (ytuple, xlist) in tqdm.tqdm(enumerate(ymap.iteritems()), total=len(yValsToXpoints)):
    # Convert ytuple back into a tensor
    yval = torch.Tensor(list(ytuple)).view(1, -1)
    logging.getLogger(__name__).info(yval.size())

    dataTensor = getGANChimeras(netG, nSamples, yval, gpu_id=gpu_id, dropoutOn=True if netG.inputSize < 1 else False)
    kde_train_input = torch.transpose(dataTensor, 0, 1).data.cpu().numpy()
    gpw = ss.gaussian_kde(kde_train_input, bw_method=bw_method)
    devData = np.concatenate(xlist, axis=1)
    probs[yval] = gpw.evaluate(devData)
    if i % logPer == (logPer - 1):
      logging.getLogger(__name__).info("number of x points: %d" % len(xlist))
      logging.getLogger(__name__).info("prob sample: %s" % repr(probs[yval]))

  return probs
