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
  def forward(self, x, y, train=False, ignoreNoise=False, ignoreCond=False):
    """
    x - the image noise
    y - the image labels, and the similarity score
    """
    # TODO
    ignoreNoise = True if self.inputSize < 1 else False
    if not ignoreNoise:
      assert(len(x) == len(y)), "invalid len(x)=%d, len(y)=%d" % (len(x), len(y))
      assert(len(x[0]) == self.inputSize), "invalid len(x[0])=%d, expect %d" % (len(x[0]), self.inputSize)
    assert(len(y[0]) == 13 or len(y[0]) == 12), "invalid len(y[0])=%d, expect 12 or 13" % (len(y[0]))
    context = y[:,:12] # Everything except the similarity score
    if ignoreCond:
      # TODO this is a bit scary; this will destroy my data in the caller? :( TODO should copy?
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

def trainCGANTest(datasetFileName = "data/models/nngandataTrain_test", ganFileName = "data/models/ganModel_test"):
  trainCGAN(datasetFileName, ganFileName)

def trainCGAN(datasetFileName = "data/models/nngandataTrain_gs_True", ganFileName = "data/models/ganModel", gpu_id=2, lr=1e-7, logPer=20, batchSize=128, epochs=5, saveDir=None, savePerEpoch=5, lam=0.5, nz=100, hiddenSize=1024, **kwargs):
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
  
  xSize = dataset[0][0].size(0)
  assert(xSize == 1024), "Invalid dataset dimensions!"

  netD = NetD(xSize, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  netG = NetG(nz, xSize, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize, gdropout).cuda(gpu_id)
  criterion = nn.BCELoss().cuda(gpu_id)
  l1Criterion = nn.L1Loss().cuda(gpu_id)
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  label = ag.Variable(torch.FloatTensor(dataloader.batch_size).cuda(gpu_id))
  # TODO for now, sgd
  #optimizerD = optim.Adam(netD.parameters(), lr=lr, betas = (beta1, 0.999))
  #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))
  optimizerD = optim.SGD(netD.parameters(), lr=lr)
  optimizerG = optim.SGD(netG.parameters(), lr=lr)
  ignoreNoise = True if nz < 1 else False

  if saveDir is not None:
    if os.path.exists(saveDir):
      logging.getLogger(__name__).info("Save directory %s already exists; cowardly exiting..." % saveDir)
      sys.exit(1)
    os.makedirs(saveDir)

  logging.getLogger(__name__).info("Beginning training")
  for epoch in tqdm.tqdm(range(epochs), total=epochs, desc="Epochs completed: "):
    for i, data in tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Iterations completed: ", leave=False):
      # Update D network
      for _ in range(dUpdates):
        netD.zero_grad()
        xdata, ydata = data

        #logging.getLogger(__name__).info("Sample x points: %s" % str(xdata[:10]))
        #logging.getLogger(__name__).info("Sample y points: %s" % str(ydata[:10]))

        xdata, ydata = ag.Variable(xdata.cuda(gpu_id)), ag.Variable(ydata.cuda(gpu_id))
        # get scores from ydata
        scores = ydata[:,12]
        output = netD(xdata, ydata, True)
        label.data.fill_(real_label)
        # TODO try adding the similarity score in here.
        errD_real = criterion(
            torch.mul(output.view(-1), scores),
            torch.mul(label[:len(output)], scores))
        errD_real.backward()
        D_x = output.data.mean()
        # D_x is how many of the images (all of which were real) were identified as real.

        noise.data.normal_(0, 1)
        fake = netG(noise[:len(ydata)], ydata, True, ignoreNoise=ignoreNoise, ignoreCond=ignoreCond)
        label.data.fill_(fake_label)
        output = netD(fake, ydata, True)
        errD_fake = criterion(
            torch.mul(output, scores),
            torch.mul(label[:len(output)], scores))
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        # D_G_z1 is how many of the images (all of which were fake) were identified as real.
        errD = errD_real + errD_fake
        optimizerD.step()

      for _ in range(gUpdates):
        # Update G network
        noise.data.normal_(0, 1)
        netG.zero_grad()
        fake = netG(noise[:len(ydata)], ydata, True, ignoreNoise=ignoreNoise, ignoreCond=ignoreCond)
        label.data.fill_(real_label)
        output = netD(fake, ydata, True)
        errG = criterion(
            torch.mul(output, scores),
            torch.mul(label[:len(output)], scores))
        errG.backward()
        D_G_z2 = output.data.mean()
        # D_G_z2 is how many of the images (all of which were fake) were identified as real, AFTER the discriminator update.

        fake = netG(noise[:len(ydata)], ydata, True, ignoreNoise=ignoreNoise, ignoreCond=ignoreCond)
        resizescores = scores.contiguous().view(-1, 1).expand(fake.size())
        errG_L1 = l1Criterion(
            torch.mul(torch.mul(fake, lam), resizescores),
            torch.mul(torch.mul(xdata, lam), resizescores))
        errG_L1.backward()
        optimizerG.step()

      if i % logPer == logPer - 1:
        tqdm.tqdm.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        logging.getLogger(__name__).info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.data[0], errG.data[0], errG_L1.data[0], D_x, D_G_z1, D_G_z2))
        
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

def evaluateGANModel(ganModelFile, datasetFileName, gpu_id=2):
  batchSize=16
  """
  Evaluate a GAN model generator's image creation ability in a simple way.
  """
  netD, netG = torch.load(ganModelFile)

  nz = netG.inputSize
  ignoreNoise = True if nz < 1 else False

  netG = netG.cuda(gpu_id)
  netD = netD.cuda(gpu_id)
  dataset = torch.load(datasetFileName)
  # This dataset needs to be a regular-style dataset, not a gan-style one.
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4)
  criterion = nn.MSELoss()
  runningTrue = 0.
  runningGimpy = 0.
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  datasetSize = 0.
  for i, data in tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Iterations completed: "):
    expectedDataSize = 12 + 3 + 1024
    assert(len(data[0][0]) == expectedDataSize), "expected len(data[0][0])=%d to be %d" % (len(data[0][0]), expectedDataSize)
    expectedYDataSize = 1024 + 1
    assert(len(data[1][0]) == expectedYDataSize), "expected len(data[0][0])=%d to be %d" % (len(data[1][0]), expectedYDataSize)
    # 1. No labels on generator. Calculate square loss.
    xdata, featuresAndScore = data
    conditionalData = xdata[:,:12].cuda(gpu_id)
    scores = featuresAndScore[:,-1].cuda(gpu_id).contiguous().view(-1, 1)
    conditionalData = torch.cat([conditionalData, scores], 1)
    noise.data.normal_(0, 1)
    output = netG(noise[:len(conditionalData)], ag.Variable(conditionalData.cuda(gpu_id)), False, ignoreNoise, ignoreCond=False)
    expectedFeatures = xdata[:,12 + 3:].cuda(gpu_id).contiguous()
    scores = scores.view(len(scores), 1).expand(output.size())

    correctLoss = criterion(
        ag.Variable(torch.mul(output.data, scores), requires_grad=False), 
        ag.Variable(torch.mul(expectedFeatures, scores), requires_grad=False))

    """
    scores = featuresAndScore[:,-1].cuda(gpu_id).contiguous().view(-1, 1)
    conditionalData = torch.zeros([len(conditionalData), 12]).cuda(gpu_id)
    conditionalData = torch.cat([conditionalData, scores], 1)
    output = netG(noise[:len(conditionalData)], ag.Variable(conditionalData.cuda(gpu_id)), False, ignoreNoise)
    scores = scores.view(len(scores), 1).expand(output.size())
    gimpyLoss = criterion(
        ag.Variable(torch.mul(output.data, scores)), 
        ag.Variable(torch.mul(expectedFeatures, scores)))
    """

    batchSize = len(conditionalData)
    datasetSize += batchSize

    runningTrue += correctLoss.data[0] * batchSize
    #runningGimpy += gimpyLoss.data[0] * batchSize
  runningTrue /= datasetSize
  #runningGimpy /= datasetSize
  logging.getLogger(__name__).info("True   Loss: %.4f" % runningTrue)
  #logging.getLogger(__name__).info("zerodY Loss: %.4f" % runningGimpy)

class GanDataLoader(object):
  def __init__(self, filename):
    self.filename = filename
    self.data = torch.load(self.filename)
    self.validate()
  def validate(self):
    if len(self.data) == 0:
      raise Exception("GanDataLoader", "No data found")
    x, y = self.data[0]
    assert(len(x) == 1024), "Invalid 'x' dimension '%d'" % len(x)
    assert(len(y) == 13), "Invalid 'y' dimension '%d'" % len(y)

# TODO I'm leaving dropout on! On purpose (nondeterministic output)
def getGANChimeras(netG, nTestPoints, yval, gpu_id=0):
  batchSize = 64
  out = []
  nz = netG.inputSize
  noise = ag.Variable(torch.FloatTensor(64, nz), requires_grad=False).cuda(gpu_id)
  yvar = ag.Variable(yval.expand(batchSize, yval.size(1))).cuda(gpu_id)
  for i in range(nTestPoints):
    noise.data.normal_(0, 1)
    if i * batchSize > nTestPoints:
      break
    out = torch.cat(out + [netG(noise, yvar, True, ignoreNoise=False, ignoreCond=False)], 0)
    out = [out]
  return out[0]


def parzenWindowCheck(ganModelFile, ganDevSet, **kwargs):
  #assert(False)
  """
  TODO: I need to find out why the values are always nan or 0.
  1. Try more samples in the PDF? might not help...
  2. Check if 'y' values are unique; check multiplicity.
  3. Or, maybe the model is just bad...
  """
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
  ignoreNoise = True if nz < 1 else False

  # map from conditioned value to a list of datapoints that match that value.
  yValsToXpoints = collections.defaultdict(list)
  # If this batchSize is not 1, we're fucked. TODO
  devDataLoader = torch.utils.data.DataLoader(gDiskDevLoader.data, batch_size=1, shuffle=True, num_workers=0)
  logging.getLogger(__name__).info("Testing dev data against parzen window fit")
  for i, data in tqdm.tqdm(enumerate(devDataLoader, 0), total=len(devDataLoader), desc="Iterations completed: "):
    #logging.getLogger(__name__).info(i)
    if i >= nTestSamples:
      #logging.getLogger(__name__).info("triggered break")
      #logging.getLogger(__name__).info("batchSize, ")
      break
    xdata, ydata = data
    ydata = ydata[:,:12] # ignore score
    # xdata is 1x1024
    kde_input = torch.transpose(xdata, 0, 1).numpy()
    #logging.getLogger(__name__).info(kde_input.shape)
    # kde_input is 1024x1
    yValsToXpoints[ydata].append(kde_input)
  #logging.getLogger(__name__).info(yValsToXpoints)

  logging.getLogger(__name__).info(len(yValsToXpoints))
  agg = collections.defaultdict(int)
  for k, v in yValsToXpoints.iteritems():
  #  # Oh wait, I bet I need to print the size or shape... TODO
    #logging.getLogger(__name__).info("samps=%s" % str(len(v)))
    agg[len(v)] += 1
  logging.getLogger(__name__).info(agg)


  #noise = ag.Variable(torch.FloatTensor(batchSize, nz), requires_grad=False).cuda(gpu_id)
  probs = {}
  for i, (yval, xlist) in tqdm.tqdm(enumerate(yValsToXpoints.iteritems()), total=len(yValsToXpoints)):
    dataTensor = getGANChimeras(netG, nSamples, yval, gpu_id=gpu_id)
    #logging.getLogger(__name__).info(dataTensor)
    kde_train_input = torch.transpose(dataTensor, 0, 1).data.cpu().numpy()
    gpw = ss.gaussian_kde(kde_train_input, bw_method=bw_method)
    devData = np.concatenate(xlist, axis=1)
    #logging.getLogger(__name__).info(devData.shape)
    probs[yval] = gpw.evaluate(devData)
    if i % logPer == (logPer - 1):
      logging.getLogger(__name__).info("number of x points: %d" % len(xlist))
      logging.getLogger(__name__).info("prob sample: %s" % repr(probs[yval]))

  logging.getLogger(__name__).info("Y data probs blah")

  return probs




  ###################### MAGINOT LINE
  """

    noise.data.normal_(0, 1)

    samples.append(netG(noise[:len(xdata)], ydata, False, ignoreNoise=ignoreNoise))

    

    logging.getLogger(__name__).info("Getting xdata")
    xdata = ag.Variable(xdata, requires_grad=False)
    logging.getLogger(__name__).info("Transposing")
    logging.getLogger(__name__).info("Calculating/appending")
    probs += list(gpw.evaluate_log(kde_input))
  # Get all 'y' samples from which to generate the distributions.


  samples = []
  # Generate samples to train the parzen window estimator
  trainDataLoader = torch.utils.data.DataLoader(gDiskTrainLoader.data, batch_size=batchSize, shuffle=True, num_workers=4)
  logging.getLogger(__name__).info("Generating Parzen window samples")
  counts = collections.defaultdict(int) # TODO
  tot = 0
  for i, data in tqdm.tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), desc="Iterations completed: "):
    #if batchSize * i >= nSamples:
    #  break
    xdata, ydata = data
    xdata, ydata = ag.Variable(xdata, requires_grad=False).cuda(gpu_id), ag.Variable(ydata, requires_grad=False).cuda(gpu_id)
    for elem in ydata:
      counts[elem] += 1
      tot += 1
    continue
    noise.data.normal_(0, 1)
    #logging.getLogger(__name__).info(xdata.size())
    #logging.getLogger(__name__).info(ydata.size())
    samples.append(netG(noise[:len(xdata)], ydata, False, ignoreNoise=ignoreNoise))
  # 'samples' is a list of tensors [N, d=1024]
  agg = collections.defaultdict(int)
  for k,v in counts.iteritems():
    agg[v] += 1
  logging.getLogger(__name__).info("GaussianParzenWindow function doesn't work. quitting...")
  import sys
  sys.exit(1)

  logging.getLogger(__name__).info("agg=%s" % str(agg))
  logging.getLogger(__name__).info("total=%d" % tot)

  samples = torch.cat(samples, 0)
  logging.getLogger(__name__).info(samples.size())
  kde_input = torch.transpose(samples, 0, 1).data.cpu().numpy()
  import utils.kde_hack as ss
  logging.getLogger(__name__).info(kde_input)
  
  logging.getLogger(__name__).info("Fitting Parzen window")
  gpw = ss.gaussian_kde(kde_input, bw_method=1e14)
  # TODO just look at his internals...
  #logging.getLogger(__name__).info(gpw(kde_input))
  #exit(1)
  #gpw(kde_input)

  # Get the dev set data
  # TODO also try the training set?
  logging.getLogger(__name__).info("A few probs: %s... " % str(probs)[:30])
  logging.getLogger(__name__).info(str(probs[0]))
  logging.getLogger(__name__).info(repr(probs[0]))
  logging.getLogger(__name__).info("Average Prob: %.8f" % (sum(probs) / (1. * len(probs))))
  """
