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
import tqdm
import sys
import os
import collections
import copy
import constants as pc # short for pair_train constants
import multiprocessing
import tblib.pickling_support
import itertools as it
tblib.pickling_support.install()

# This is used for capturing errors from subprocesses
# See rocksportrocker comment here:
# http://stackoverflow.com/questions/6126007/python-getting-a-traceback-from-a-
#     multiprocessing-process/16618842
class ExceptionWrapper(object):
    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)
        # for Python 2 replace the previous line by:
        # raise self.ee, None, self.tb


class NetD(nn.Module):
  def __init__(self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs, vrESize):
    super(NetD, self).__init__()
    ydataLength = 6 * wESize + 6 * vrESize
    self.input_layer = nn.Linear(inputSize + ydataLength, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)

    self.dropout = 0.

    logging.getLogger(__name__).info("Building Discriminator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (inputSize + ydataLength))
  def forward(self, conditionals, testImage, train=False):
    """
    conditionals - the image labels, the transformed role, the first noun, the 
                   second noun, and the input image features
    testImage - the image features, and (optionally) the scores (unused)
    """
    assert(conditionals.size(0) == testImage.size(0))
    assert(conditionals.size(1) == 12 + 3 + 1024), \
        "Invalid conditionals.size(1)=%d" % conditionals.size(1)
    assert(testImage.size(1) in (1024, 1025)), \
        "Invalid testImage.size(1)=%d" % testImage.size(1)

    conditionals = conditionals[:,:12] # Everything except the similarity score
    context_vectors = pairnn.getContextVectors(
        self.contextVREmbedding, self.contextWordEmbedding, conditionals,
        testImage.size(0))
    testImage = testImage[:,:1024] # Strip off the scores.
    netInput = torch.cat([testImage] + context_vectors, 1)

    x = F.dropout(
        F.leaky_relu(
            self.input_layer(netInput)), p=self.dropout, training=train)
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
  def __init__(
      self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs,
      vrESize, dropout=0.):
    super(NetG, self).__init__()
    ydataLength = 6 * wESize + 6 * vrESize
    self.inputSize = inputSize
    self.input_layer = nn.Linear(self.inputSize + ydataLength, hiddenSize)
    self.hidden_layers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)
    self.dropout = dropout

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)
    logging.getLogger(__name__).info("Building Generator network")
    logging.getLogger(__name__).info("--> Input size: %d" % (self.inputSize + ydataLength))
  def forward(self, z, conditionals, train=False, ignoreCond=False): # TODO pick up here.
    """
    Generate an image that matches the given conditionals
    z - the input noise (possibly empty)
    conditionals - the image labels, the transformed role, the first noun, the 
                   second noun, and the input image features
    """
    # Ignore everything except the input labels.
    ignoreNoise = True if self.inputSize < 1 else False

    assert(z.size(0) == conditionals.size(0)), "batch size mismatch"
    if not ignoreNoise:
      assert(z.size(1) == self.inputSize), "invalid z.size(1)=%d" % z.size(1)
    assert(conditionals.size(1) == 12 + 3 + 1024), \
        "invalid conditionals.size(1)=%d" % conditionals.size(1)

    context = conditionals[:,:12] # Everything except the similarity score
    if ignoreCond:
      context = context.clone()
      context.data.fill_(0)

    context_vectors = pairnn.getContextVectors(self.contextVREmbedding, self.contextWordEmbedding, context, z.size(0))
    toCat = context_vectors
    if not ignoreNoise:
      toCat = [x] + toCat
    netInput = torch.cat(toCat, 1)

    x = F.dropout(
        F.leaky_relu(
            self.input_layer(netInput)), training=train, p=self.dropout) 
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(
          F.leaky_relu(
              self.hidden_layers[i](x)), training=train, p=self.dropout)
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
  raise ValueError(
      "Invalid data set with xdata[0] dim=%d, ydata[0] dim=%d" % \
          (len(xdata[0]), len(ydata[0])))

class TrGanD(nn.Module):
  def __init__(
      self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs,
      vrESize):
    super(TrGanD, self).__init__()
    self.ydataLength = (6 * wESize + 6 * vrESize) \
                       + (vrESize) + (2 * wESize) \
                       + pc.IMFEATS
    self.input_layer = nn.Linear(inputSize + self.ydataLength, hiddenSize)
    self.hidden_layers = nn.ModuleList(
        [nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)

    self.dropout = 0.

    logging.getLogger(__name__).info("Building Discriminator network")
    logging.getLogger(__name__).info(
        "--> Input size: %d" % (inputSize + self.ydataLength))
  def forward(self, conditionals, testImage, train=False):
    """
    Determine whether the input testImage is real or fake, given the information
    inside "conditionals"
    conditionals - the image labels, the transformed role, the first noun, the 
                   second noun, and the input image features
    testImage - the image features, and (optionally) the scores (unused)
    """
    assert(testImage.size(0) == conditionals.size(0))
    assert(testImage.size(1) in (1024, 1025)), \
        "Invalid testImage.size(1)=%d" % testImage.size(1)
    assert(conditionals.size(1) == 12 + 3 + pc.IMFEATS), \
        "Invalid conditionals.size(1)=%d" % conditionals.size(1)

    testImage = testImage[:,:pc.IMFEATS] # get rid of the scores.

    batchSize = testImage.size(0)
    context = getTrganContext(
        batchSize, self.contextVREmbedding, self.contextWordEmbedding,
        conditionals)
    netInput = torch.cat([testImage, context], 1)

    x = F.dropout(
        F.leaky_relu(
            self.input_layer(netInput)), p=self.dropout, training=train)
    for i in range(0, len(self.hidden_layers)):
      x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train)
      if i == 0: x_prev = x
      #add skip connections for depth
      if i > 0 and i % 2 == 0: 
        x = x + x_prev
        x_prev = x
    x = F.sigmoid(self.output(x))
    return x

def getTrganContext(
    batchsize, contextVREmbedding, contextWordEmbedding, conditionals):
  annotations = pairnn.getContextVectors(
      contextVREmbedding, contextWordEmbedding, conditionals[:,:12], batchsize)
  role = contextVREmbedding(conditionals[:,12].long()).view(batchsize, -1)
  n1 = contextWordEmbedding(conditionals[:,13].long()).view(batchsize, -1)
  n2 = contextWordEmbedding(conditionals[:,14].long()).view(batchsize, -1)
  img = conditionals[:,15:15 + pc.IMFEATS]
  context = torch.cat(annotations + [role, n1, n2, img], 1)
  return context

class TrGanG(nn.Module):
  def __init__(
      self, inputSize, outputSize, hiddenSize, depth, nWords, wESize, nVRs,
      vrESize, dropout=0.):
    super(TrGanG, self).__init__()
    self.ydataLength = (6 * wESize + 6 * vrESize) + (vrESize) + (2 * wESize) + \
        pc.IMFEATS
    self.inputSize = inputSize

    self.input_layer = nn.Linear(self.inputSize + self.ydataLength, hiddenSize)
    self.hidden_layers = nn.ModuleList(
        [nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)
    self.dropout = dropout

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)
    logging.getLogger(__name__).info("Building Generator network")
    logging.getLogger(__name__).info(
        "--> Input size: %d" % (self.inputSize + self.ydataLength))
    # TODO should I use padding_idx on the embeddings, to make sure that 
    # removing the conditional corresponds to the correct representation?
  def forward(self, z, conditionals, train=False, ignoreCond=False):
    """
    Generate an image that matches the given conditionals
    z - the input noise (possibly empty)
    conditionals - the image labels, the transformed role, the first noun, the 
                   second noun, and the input image features
    """
    # TODO @mchorton I ignored att_w, att_r, att_b from Mark's NN code.
    ignoreNoise = True if self.inputSize < 1 else False
    if not ignoreNoise:
      assert(len(z[0]) == self.inputSize), \
          "invalid len(z[0])=%d, expect %d" % (len(z[0]), self.inputSize)
    assert(len(conditionals[0]) == (12 + 3 + pc.IMFEATS)), \
        "invalid len(conditionals[0])=%d, expect %d" % \
            (len(conditionals[0]), 12 + 3 + pc.IMFEATS)
    assert(len(z) == len(conditionals)), \
        "invalid len(z)=%d, len(conditionals)=%d" % (len(z), len(conditionals))

    batchsize = len(z)
    context = getTrganContext(
        batchsize, self.contextVREmbedding, self.contextWordEmbedding, 
        conditionals)

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

class GanTrainer(object):
  def __init__(self, **kwargs):
    self.kwargs = kwargs
  def train(self, datasetFileName, ganFileName, **moreKwargs):
    kwargs = copy.deepcopy(self.kwargs)
    kwargs.update(moreKwargs)
    trainCGAN(
        datasetFileName=datasetFileName, ganFileName=ganFileName, **kwargs)

def saveShard(outFileName, datalist):
  """
  Create a Tensor Dataset from the list of pairs in datalist, then save to 
  outFileName
  """
  xdata, ydata = zip(*datalist)
  xdata = torch.cat(xdata, 0)
  ydata = torch.cat(ydata, 0)
  assert(xdata.size(0) == ydata.size(0)), "Batch size mismatch"
  assert(xdata.size(1) == 12 + 3 + 1024), "Invalid x.size(1)=%d" % xdata.size(1)
  assert(ydata.size(1) == 1024 + 1), "Invalid y.size(1)=%d" % ydata.size(1)
  torch.save(
      td.TensorDataset(xdata, ydata), outFileName)

class ShardedDataHandler(object):
  # Class for reading the folder output of shardAndSave
  def __init__(self, directory):
    self.directory = directory
    if not os.path.exists(self.directory):
      os.makedirs(self.directory)
  @staticmethod
  def keyToName(key):
    return "_".join(map(str, key)) + ".data"
  def save(self, datashards):
    """
    Save a bunch of datasets to disk.
    datashards - map from (n1, n2) integer IDs -> [data, data, data, ...]
    here, 'data' is a x,y pair (batch size one, e.g. x.size() == (1, 12+3+1024))
    """
    for k,datalist in tqdm.tqdm(
        datashards.iteritems(), total=len(datashards), desc="Saving datasets"):
      outname = os.path.join(self.directory, self.keyToName(k))
      saveShard(outname, datalist)

def histString(histogram):
    """
    Represent a np.histogram in a nice / readable way.
    """
    hist, binEdges = histogram
    del histogram
    ret = "[" + "<%s>" % str(binEdges[0])
    binEdges = binEdges[1:]
    for val, edge in it.izip(hist, binEdges):
        ret += " %s <%s>" % (val, edge)
    return ret + ")"

def shardAndSave(datasetFileName, outDirName, **kwargs):
  """
  Break up data in datasetFileName based on noun pairs. Save the chunks of data
  to folder outDirName.
  """
  minDataPts = kwargs.get("minDataPts", 0)
  # TODO assert that this dataset is in the right form?
  nounpairToData = collections.defaultdict(list)
  dataSet = torch.load(datasetFileName)
  dataloader = td.DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=0)
  for i, data in tqdm.tqdm(
      enumerate(dataloader, 0), total=len(dataloader),
      desc="Sharding data...", leave=False):
    conditionals, testImageAndScore = data
    key = (conditionals[(0, 13)], conditionals[(0, 14)]) # Get n1, n2
    nounpairToData[key].append((data[0].clone(), data[1].clone()))
  logging.getLogger(__name__).info(
      "Got %d unique noun pairs" % len(nounpairToData))
  bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 10000, sys.maxint]
  distribution = np.histogram(
      [len(v) for v in nounpairToData.itervalues()], bins=bins)
  logging.getLogger(__name__).info(
      "Distribution of dataset sizes: %s" % histString(distribution))
  logging.getLogger(__name__).info(
      "Requiring at least %d datapoints" % minDataPts)
  nounpairToData = {
      pair: data for pair, data in nounpairToData.iteritems() if len(data) > minDataPts}
  logging.getLogger(__name__).info(
      "Filtered down to %d unique noun pairs" % len(nounpairToData))
  saver = ShardedDataHandler(outDirName)
  saver.save(nounpairToData)

def trainOneGanArgstyle(args):
  trainOneGan(*args)
def trainOneGan(datasetFileName, ganFileName, logFileName, gpuId, ganTrainer):
  ganTrainer.train(
      datasetFileName, ganFileName, logFileName=logFileName, gpu_id=gpuId)

def trainIndividualGans(datasetDirectory, ganDirectory, logFileDirectory, **kwargs):
  # TODO nice to do something fancier. For now, just shard jobs between these 
  # GPUs
  ganTrainer = GanTrainer(**kwargs)
  procsPerGpu = kwargs.get("procsPerGpu", 3)
  activeGpus = [0, 1, 2, 3]
  nGpus = len(activeGpus)
  gpuToTasks = {gpu: [] for gpu in activeGpus}
  for i, filename in enumerate(os.listdir(datasetDirectory)):
    filenameNoExt = os.path.splitext(filename)[0]
    gpuId = i % nGpus
    gpuToTasks[gpuId].append(
        (
            os.path.join(datasetDirectory, filename),
            os.path.join(
                ganDirectory, 
                filenameNoExt + ".gan"),
            os.path.join(
                logFileDirectory, filenameNoExt + ".log"),
            gpuId,
            ganTrainer))
  pools = {}
  for gpu in activeGpus:
    pools[gpu] = multiprocessing.Pool(processes=procsPerGpu)
  resultmap = {}
  for gpu, pool in pools.iteritems():
    resultmap[gpu] = pool.map_async(trainOneGanArgstyle, gpuToTasks[gpu])
    pool.close()

  for gpu, pool in pools.iteritems():
    pool.join()

  for k,v in resultmap.iteritems():
    finalResults = v.get()
    for result in finalResults:
      if isinstance(result, ExceptionWrapper):
        result.re_raise()
  logging.getLogger(__name__).info(resultmap)

def genDataAndTrainIndividualGansStubTest():
  genDataAndTrainIndividualGans(
      "data/multigan_test/",
      "data/pairLearn/comptrain_test.pyt_mode_max",
      "data/runs/test_runs/",
      epochs=2,
      logPer=1,
      genDepth=2,
      depth=2,
      procsPerGpu=1,
      minDataPts=5,
      style="trgan")

def genDataAndTrainIndividualGansStub():
  genDataAndTrainIndividualGans(
      "data/multigan_2/",
      "data/pairLearn/comptrain.pyt_mode_max",
      "data/runs/multigan_2/",
      epochs=200,
      logPer=3,
      genDepth=32,
      depth=32,
      procsPerGpu=4,
      lr=1e-2,
      minDataPts=100,
      decayPer=10,
      decayRate=0.7,
      batchSize=32
      style="trgan")

def genDataAndTrainIndividualGans(
    outDirName, datasetFileName, logFileDirectory, **kwargs):
  datasetDir = os.path.join(outDirName, "nounpair_data/")
  logging.getLogger(__name__).info("kwargs: %s" % str(kwargs))
  shardAndSave(datasetFileName, datasetDir, **kwargs)
  ganLoc = os.path.join(outDirName, "trained_models/")
  if not os.path.exists(ganLoc):
    os.makedirs(ganLoc)
  trainIndividualGans(
      datasetDir, ganLoc, logFileDirectory, mapBaseName=datasetFileName,
      **kwargs)

def trainCGAN(
    datasetFileName = "data/models/nngandataTrain_gs_True",
    ganFileName = "data/models/ganModel", gpu_id=0, lr=1e-2, logPer=5,
    batchSize=128, epochs=5, saveDir=None, savePerEpoch=5, lam=1e-2, nz=0,
    hiddenSize=pc.IMFEATS, **kwargs):
  # Set up variables.
  beta1=0.9999
  mapBaseName = kwargs.get("mapBaseName", datasetFileName)
  role2Int = torch.load("%s_role2Int" % mapBaseName)
  noun2Int = torch.load("%s_noun2Int" % mapBaseName)
  verb2Len = torch.load("%s_verb2Len" % mapBaseName)
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
  style = kwargs.get("style", "trgan")
  # Whether to ignore the conditioning variable. Useful for baselines.
  ignoreCond = kwargs.get("ignoreCond", False) 
  logFileName = kwargs.get("logFileName", None)
  # TODO sanitize for unused variables.
  # Load variables, begin training.
  dataset = torch.load(datasetFileName)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)
  
  if style == "gan":
    netG = NetG(nz, pc.IMFEATS, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize, gdropout).cuda(gpu_id)
    netD = NetD(pc.IMFEATS, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  elif style == "trgan":
    netG = TrGanG(nz, pc.IMFEATS, hiddenSize, genDepth, nWords, wESize, nVRs, vrESize, gdropout).cuda(gpu_id)
    netD = TrGanD(pc.IMFEATS, 1, hiddenSize, depth, nWords, wESize, nVRs, vrESize).cuda(gpu_id)
  else:
    raise ValueError("Invalid style '%s'" % style)
  criterion = nn.BCELoss().cuda(gpu_id)
  l1Criterion = nn.L1Loss().cuda(gpu_id)
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  label = ag.Variable(torch.FloatTensor(dataloader.batch_size).cuda(gpu_id))
  optimizerD = optim.SGD(netD.parameters(), lr=lr)
  optimizerG = optim.SGD(netG.parameters(), lr=lr)

  if saveDir is not None:
    if os.path.exists(saveDir):
      logging.getLogger(__name__).info(
          "Save directory %s already exists; cowardly exiting..." % saveDir)
      sys.exit(1)
    os.makedirs(saveDir)

  with open(logFileName, "w") as logFile:
    for epoch in tqdm.tqdm(
        range(epochs), total=epochs, desc="Epochs completed: "):
      for i, data in tqdm.tqdm(
          enumerate(dataloader, 0), total=len(dataloader),
          desc="Iterations completed: ", leave=False):
        networkUpdateStep(
            netD, netG, data, dUpdates, gUpdates, noise, label, gpu_id,
            ignoreCond, i, style, optimizerD, optimizerG, criterion, l1Criterion,
            logPer, lam, epoch, epochs, len(dataloader), logFile)
          
      # save the training files
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

# TODO turn this into a class.
def networkUpdateStep(
    netD, netG, data, dUpdates, gUpdates, noise, label, gpu_id, ignoreCond, i, 
    style, optimizerD, optimizerG, criterion, l1Criterion, logPer, lam, epoch,
    epochs, nSamples, logFile):
  real_label = 1
  fake_label = 0
  # Update D network
  for _ in range(dUpdates):
    netD.zero_grad()
    conditional, realImageAndScore = data
    conditional, realImageAndScore = ag.Variable(conditional.cuda(gpu_id)), \
        ag.Variable(realImageAndScore.cuda(gpu_id))

    # get scores from realImageAndScore
    scores = realImageAndScore[:,pc.IMFEATS]
    output = netD(conditional, realImageAndScore, True)
    label.data.fill_(real_label)
    errD_real = criterion(
        torch.mul(output.view(-1), scores),
        torch.mul(label[:len(output)], scores))
    errD_real.backward()
    # D_x is how many of the images (all of which were real) were identified as
    # real.
    D_x = output.data.mean()

    noise.data.normal_(0, 1)
    fake = netG(
        noise[:len(realImageAndScore)], conditional, True,
        ignoreCond=ignoreCond)
    label.data.fill_(fake_label)
    output = netD(conditional, fake, True)
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
    fake = netG(
        noise[:len(realImageAndScore)], conditional, True,
        ignoreCond=ignoreCond)
    label.data.fill_(real_label)
    output = netD(conditional, fake, True)
    errG = criterion(
        torch.mul(output, scores),
        torch.mul(label[:len(output)], scores))
    errG.backward()
    D_G_z2 = output.data.mean()
    # D_G_z2 is how many of the images (all of which were fake) were identified
    # as real, AFTER the discriminator update.

    fake = netG(
        noise[:len(realImageAndScore)], conditional, True,
        ignoreCond=ignoreCond)
    resizescores = scores.contiguous().view(-1, 1).expand(fake.size())
    errG_L1 = l1Criterion(
        torch.mul(torch.mul(fake, lam), resizescores),
        torch.mul(torch.mul(realImageAndScore[:,:1024], lam), resizescores))
    errG_L1.backward()
    optimizerG.step()

  if i % logPer == logPer - 1:
    logString = \
        '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' \
              % (epoch, epochs, i, nSamples, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2)
    if logFile:
      logFile.write(logString + '\n')

    tqdm.tqdm.write(logString)
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
      dataset, batch_size=batchSize, shuffle=True, num_workers=0)
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

def parzenWindowFromFile(ganModelFile, ganDevSet, **kwargs):
  gDiskDevLoader = GanDataLoader(ganDevSet)
  _, netG = torch.load(ganModelFile)
  parzenWindowProb(netG, gDiskDevLoader.data, **kwargs)

def parzenWindowProb(netG, devData, **kwargs):
  # TODO this is using ydata as the conditional! It shouldn't.
  # I need to fix many things about this function.
  """
  nSamples - number of points to draw from generator when making distribution
  nTestSamples - number of points to draw from devData for testing
  """
  # Set up some values.
  gpu_id = kwargs.get("gpu_id", 2)
  logPer = kwargs.get("logPer", 1e12)
  bw_method = kwargs.get("bw_method", "scott")
  nSamples = kwargs.get("nSamples", 100)
  nTestSamples = kwargs.get("nTestSamples", 100)

  netG = netG.cuda(gpu_id)

  # map from conditioned value to a list of datapoints that match that value.
  ymap = collections.defaultdict(list)

  # This batch_size must be one, because I don't iterate over ydata again.
  devDataLoader = torch.utils.data.DataLoader(
      devData, batch_size=1, shuffle=False, num_workers=0)
  logging.getLogger(__name__).info("Testing dev data against parzen window fit")
  for i, data in tqdm.tqdm(
      enumerate(devDataLoader, 0), total=len(devDataLoader),
      desc="Iterations completed: "):
    if i >= nTestSamples:
      break
    conditional, imageAndScore = data
    image = imageAndScore[:,:1024]
    if style == "gan":
      conditional = conditional[:,:12]
    elif syle == "trgan":
      pass
    else:
      raise ValueError("Invalid style '%s'" % style)
    # kde_input is pc.IMFEATSx1
    kde_input = torch.transpose(xdata, 0, 1).numpy()
    key = tuple(*ydata.numpy().tolist())
    ymap[key].append(kde_input)

  probs = {}
  for i, (ytuple, xlist) in tqdm.tqdm(
      enumerate(ymap.iteritems()), total=len(ymap)):
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
