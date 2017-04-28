# Training the image transformation logic using a neural network.
import json
import hack.stats.kde as kde
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
import itertools as it
import pair_train.dashboard as dash
import subprocess
import utils.methods as mt
import data as dataman
import pandas as pd
import utils.plotutils as pu
from traceback import print_exc

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
  def forward(self, z, conditionals, train=False, ignoreCond=False):
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
      vrESize, noImg):
    super(TrGanD, self).__init__()
    self.ydataLength = (6 * wESize + 6 * vrESize) \
                       + (vrESize) + (2 * wESize)
    if not noImg:
      self.ydataLength += pc.IMFEATS
    self.input_layer = nn.Linear(inputSize + self.ydataLength, hiddenSize)
    self.hidden_layers = nn.ModuleList(
        [nn.Linear(hiddenSize, hiddenSize) for i in range(depth)])
    self.output = nn.Linear(hiddenSize, outputSize)

    self.contextWordEmbedding = nn.Embedding(nWords+1, wESize)
    self.contextVREmbedding = nn.Embedding(nVRs+1, vrESize)

    self.dropout = 0.
    self.noImg = noImg

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
        conditionals, self.noImg)
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
    batchsize, contextVREmbedding, contextWordEmbedding, conditionals, noImg=False):
  annotations = pairnn.getContextVectors(
      contextVREmbedding, contextWordEmbedding, conditionals[:,:12], batchsize)
  role = contextVREmbedding(conditionals[:,12].long()).view(batchsize, -1)
  n1 = contextWordEmbedding(conditionals[:,13].long()).view(batchsize, -1)
  n2 = contextWordEmbedding(conditionals[:,14].long()).view(batchsize, -1)
  toCat = annotations + [role, n1, n2]
  if not noImg:
    toCat.append(conditionals[:,15:15 + pc.IMFEATS])
  context = torch.cat(toCat, 1)
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
  trainCGAN(ganFileName, datasetFileName)

# Can I get rid of this in favor of CheckpointTrainer? TODO
class GanTrainer(object):
  def __init__(self, **kwargs):
    self.kwargs = kwargs
  def train(self, datasetFileName, devDatasetFileName, pairDataManager, ganFileName, **moreKwargs):
    kwargs = copy.deepcopy(self.kwargs)
    kwargs.update(moreKwargs)
    trainCGAN(
        ganFileName, datasetFileName, devDatasetFileName, pairDataManager,
        **kwargs)
  def trainNoFile(self, datasetFileName, devDatasetFileName, pairDataManager, **moreKwargs):
    kwargs = copy.deepcopy(self.kwargs)
    kwargs.update(moreKwargs)
    return trainCGANNoFile(
        datasetFileName, devDatasetFileName, pairDataManager, **kwargs)

def redirectOutputAndTrain(args): 
  pid = os.getpid()
  logFileName = args[3]
  mt.setOutputToFiles(logFileName)
  logging.getLogger(__name__).info("Training gan in process %d" % pid)
  logging.getLogger(__name__).info("Using gpu %d" % args[4])
  try:
    logging.getLogger(__name__).info("Training...")
    trainOneGanArgstyle(args)
  except Exception as e:
    logging.getLogger(__name__).info("Printing Exception...")
    print_exc(1000)
    logging.getLogger(__name__).info("Done Printing Exception...")
    return e
  return "Completed process %d" % pid

def trainOneGanArgstyle(args):
  trainOneGan(*args)

# TODO why not kwargs?
def trainOneGan(
    ganFileName, datasetFileName, devDatasetFileName, logFileName, gpu_id,
    ganTrainer, parzenDirectory, ablationDirectory, useScore, pairtraindir,
    featdir, kwargs):
  measurePerf = kwargs.get("measurePerf", False)
  logging.getLogger(__name__).info("measurePerf=%s" % measurePerf)
  with mt.perfmeasure(measurePerf, logging.getLogger(__name__).info) as pm:
    pairDataManager = dataman.PairDataManager(pairtraindir, featdir)
    logging.getLogger(__name__).info("Training GAN")
    logging.getLogger(__name__).info("gpu_id is %d" % gpu_id)
    logging.getLogger(__name__).info("kwargs is %s" % str(kwargs))
    ganTrainer.train(
        datasetFileName, devDatasetFileName, pairDataManager, ganFileName,
        useScore=useScore, logFileName=logFileName,  gpu_id=gpu_id,
        trainGraphBn=os.path.splitext(logFileName)[0])

def trainIndividualGans(datasetDirectory, devsetDirectory, ganDirectory, logFileDirectory, parzenDirectory, ablationDirectory, pairtraindir, featdir, **kwargs):
  ganTrainer = GanTrainer(**kwargs)
  procsPerGpu = kwargs.get("procsPerGpu", 3)
  seqOverride = kwargs.get("seqOverride", False)
  activeGpus = kwargs["activeGpus"]
  nGpus = len(activeGpus)
  taskList = []
  for i, filename in enumerate(os.listdir(datasetDirectory)):
    filenameNoExt = os.path.splitext(filename)[0]
    gpu_id = activeGpus[i % nGpus]
    taskList.append(
        (
            os.path.join(
                ganDirectory, 
                filenameNoExt + ".gan"),
            os.path.join(datasetDirectory, filename),
            os.path.join(devsetDirectory, filename),
            os.path.join(
                logFileDirectory, filenameNoExt + ".log"),
            gpu_id,
            ganTrainer,
            parzenDirectory,
            ablationDirectory,
            kwargs["useScore"],
            pairtraindir,
            featdir,
            kwargs))

  if kwargs.get("onlyN", None) and len(taskList) > 0:
    taskList = taskList[:kwargs["onlyN"]]

  if not seqOverride:
    gpuResources = [k for _ in xrange(procsPerGpu) for k in activeGpus]
    pool = multiprocessing.Pool(processes=len(gpuResources), maxtasksperchild=1)
    pending = []

    with open(
        os.path.join(logFileDirectory, "multiproc_log.log"), "w+") as mlog:

      for task in tqdm.tqdm(taskList, total=len(taskList), desc="Dispatching jobs"):
        while not gpuResources:
          mlog.write("Querying ready tasks\n")
          ready = []
          notready = []
          for p, gpu_id in pending:
            p.wait(1) # TODO without this, it doesn't work? It should...
            if p.ready():
              ready.append((p, gpu_id))
              mlog.write("Process ready. Getting result...")
              result = p.get()
              if isinstance(result, Exception):
                logging.getLogger(__name__).info("Completed process raised exception: %s" % str(result))
              mlog.write("Process result: %s" % str(result))
            else:
              notready.append((p, gpu_id))
          mlog.write("Ready: %s\n" % str(ready))
          mlog.write("NotReady: %s\n" % str(notready))
          
          for p, gpu_id in ready:
            gpuResources.append(gpu_id)
          del ready
          pending = notready
          mlog.write("Sleeping\n")
          import time; time.sleep(1)
        mlog.write("Resources is %s\n" % gpuResources)
        gpu_id = gpuResources.pop()
        t = list(task)
        t[4] = gpu_id # TODO use kwargs.
        task = tuple(t)
        mlog.write("Starting task on gpu_id=%d\n" % gpu_id)
        mlog.write("task is %s\n" % str(task))
        pending.append((pool.apply_async(redirectOutputAndTrain, (task,)), gpu_id))
      # TODO we now need to wait for the pending tasks.
      logging.getLogger(__name__).info("Waiting for remaining tasks to finish.")
      for p, gpu_id in tqdm.tqdm(pending, total=len(pending), desc="Finishing remaining tasks"):
        p.wait()
        result = p.get()
        if isinstance(result, Exception):
          logging.getLogger(__name__).info("Completed process raised exception: %s" % str(result))
        mlog.write("Process result: %s" % str(p))

  else:
    for task in taskList:
      trainOneGanArgstyle(task)

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
      style="trgan",
      gdroupout=0.05,
      useScore=False,
      seqOverride=True)

def genDataAndTrainIndividualGansStub():
  genDataAndTrainIndividualGans(
      "data/multigan_2/",
      "data/pairLearn/comptrain.pyt_mode_max",
      "data/runs/multigan_2/",
      epochs=200,
      logPer=3,
      genDepth=32,
      depth=32,
      procsPerGpu=1,
      lr=1e-2,
      lam=1e-2,
      minDataPts=50,
      decayPer=10,
      decayRate=0.7,
      batchSize=32,
      gdropout=0.05,
      useScore=False,
      style="trgan")

def exp3():
  genDataAndTrainIndividualGans(
      "data/multigan_3/",
      "data/pairLearn/comptrain.pyt_mode_max",
      "data/runs/multigan_3/",
      epochs=200,
      logPer=3,
      genDepth=32,
      depth=32,
      procsPerGpu=1,
      lr=1e-4,
      lam=1e-4,
      minDataPts=50,
      decayPer=10,
      decayRate=0.7,
      batchSize=32,
      gdropout=0.05,
      useScore=False,
      style="trgan")

def genDataAndTrainIndividualGans(
    outDirName, datasetFileName, logFileDir, pairtraindir, featdir, **kwargs):
  mt.makeDirIfNeeded(logFileDir)
  with open(os.path.join(logFileDir, "args.txt"), "w+") as argfile:
    argfile.write("ARGS=%s\nKWARGS=%s" % (str((outDirName, datasetFileName, logFileDir)), str(kwargs)))
  datasetDir = os.path.join(outDirName, "nounpair_data/")
  devsetDir = os.path.join(outDirName, "nounpair_data_dev/")
  pdm = dataman.PairDataManager(pairtraindir, featdir)
  if not kwargs.get("skipShardAndSave", False):
    dataman.shardAndSave(pdm, datasetDir, "train", **kwargs)
    dataman.shardAndSave(pdm, devsetDir, "dev", **kwargs)

  parzenDir = os.path.join(outDirName, "parzen_fits/")
  ablationDir = os.path.join(outDirName, "ablations/")
  gdm = dash.GanDashboardMaker()
  gdm.makeDashboard(
      parzenDir, logFileDir, ablationDir, datasetDir,
      datasetFileName + "_noun2Int",
      os.path.join(logFileDir, "index.php"))

  ganLoc = os.path.join(outDirName, "trained_models/")
  if not os.path.exists(ganLoc):
    os.makedirs(ganLoc)
  trainIndividualGans(
          datasetDir, devsetDir, ganLoc, logFileDir, parzenDir, ablationDir, pairtraindir,
          featdir, mapBaseName=datasetFileName, **kwargs)

def trainCGAN(ganFileName, *args, **kwargs):
  nets = trainCGANNoFile(*args, ganSaveBasename = ganFileName, **kwargs)
  logging.getLogger(__name__).info("Saving (netD, netG) to %s" % ganFileName)
  torch.save(nets, ganFileName)

class TrainGraphGenerator(object):
    def __init__(self, pairDataManager, graphbn, **kwargs):
        self._kwargs = kwargs
        self._graphbn = graphbn
        self._data = pd.DataFrame()
        self._pdm = pairDataManager
        if self._graphbn is None:
            return
        self._graphconfig = [
                {
                        "chosen": ["Loss_D", "Loss_G", "Loss_L1"],
                        "loc": self._graphbn + ".stackloss.jpg",
                        "title": "Losses",
                        "xlabel": "Epoch",
                        "ylabel": "Loss",
                        "maker": pu.makeStackplotDefault},
                {
                        "chosen": ["Dev Loss_D", "Dev Loss_G", "Dev Loss_L1"],
                        "loc": self._graphbn + ".valloss.jpg",
                        "title": "Losses",
                        "xlabel": "Epoch",
                        "ylabel": "Loss",
                        "maker": pu.makeStackplotDefault},
                {
                        "chosen": ["Parzen Average"],
                        "loc": self._graphbn + ".parzen.jpg",
                        "title": "Parzen Window Fit",
                        "xlabel": "Epoch",
                        "ylabel": "Log Prob"},
                {
                        "chosen": ["Dev Parzen Average"],
                        "loc": self._graphbn + ".devparzen.jpg",
                        "title": "Dev Parzen Window Fit",
                        "xlabel": "Epoch",
                        "ylabel": "Log Prob"},
                {
                        "chosen": ["D(x)", "D_G_z1"],
                        "loc": self._graphbn + ".predictions.jpg",
                        "title": "Disc. Prediction on Real / Fake",
                        "xlabel": "Epoch",
                        "ylabel": "Prediction (1=Real, 0=Fake)"},
                {
                        "chosen": ["Dev D(x)", "Dev D_G_z1"],
                        "loc": self._graphbn + ".valpredictions.jpg",
                        "title": "Dev Set Prediction on Real / Fake",
                        "xlabel": "Epoch",
                        "ylabel": "Prediction (1=Real, 0=Fake)"},
                {
                        "chosen": ["D_G_z1", "D_G_z2"],
                        "loc": self._graphbn + ".dgz.jpg",
                        "title": "Change in Predictions for Fake",
                        "xlabel": "Epoch",
                        "ylabel": "Prediction (1=Real, 0=Fake)"},
                {
                        "chosen": ["Loss_L1"],
                        "loc": self._graphbn + ".l1loss.jpg",
                        "title": "L1 Loss",
                        "xlabel": "Epoch",
                        "ylabel": "Loss"},
                {
                        "chosen": ["Dev Loss_L1"],
                        "loc": self._graphbn + ".devl1loss.jpg",
                        "title": "Dev L1 Loss",
                        "xlabel": "Epoch",
                        "ylabel": "Loss"},
                {
                        "chosen": ["True Loss", "Ablated Loss"],
                        "loc": self._graphbn + ".abl.jpg",
                        "title": "Ablation Losses",
                        "xlabel": "Epoch",
                        "ylabel": "Loss"},
                {
                        "chosen": ["Val True Loss", "Val Ablated Loss"],
                        "loc": self._graphbn + ".devabl.jpg",
                        "title": "Ablation Losses",
                        "xlabel": "Epoch",
                        "ylabel": "Loss"}]

        self.ablCkpt = None
        self._iterctr = 0

    def logInitialState(
            self, dataset_filename, devdataset_filename, totit, stepper):
        if self._graphbn is None:
            return
        updates = {}
        logging.getLogger(__name__).info("Getting initial state")
        self._iterctr = -1 # generate() will increase this to its proper value 0
        logging.getLogger(__name__).info("Getting training losses")
        dataset = torch.load(dataset_filename)
        updates.update(stepper.getLoss(dataset, ""))
        # pretending totit is 0 to trigger certain updates. JK, I don't have to.
        self.generate(
                stepper, 0, 0, totit, updates, dataset_filename,
                devdataset_filename, forceCalc=True)
        logging.getLogger(__name__).info("After LIT, data is %s" % str(self._data))
        
    def generate(
            self, stepper, epoch, iteration, totit, losses, datasetFileName,
            devdataset_filename, forceCalc=False):
            # iteration - the number of iterations completed this epoch (never
            # will be '0', at most will be totit)
        if self._graphbn is None:
            return
        self._iterctr += 1
        updates = self._calculateUpdates(
                stepper, epoch, iteration, totit, losses, datasetFileName,
                devdataset_filename, force=forceCalc)

        frac = float(iteration) / float(totit) if totit is not 0 else 0
        timestamp = frac + float(epoch)
        self._appendValues(timestamp, updates)
        if mt.shouldDo(self._iterctr, self._kwargs["graphPerIter"]):
            logging.getLogger(__name__).info("Making graphs")
            self._makeGraphs()

    def getAblCkpt(self, datasetFileName):
        if self.ablCkpt is None:
            ablkwargs = copy.deepcopy(self._kwargs)
            ablkwargs["ganSaveBasename"] = None
            ablkwargs["ignoreCond"] = True
            ablkwargs["trainGraphBn"] = None
            self.ablCkpt = CheckpointTrainer(
                    datasetFileName, None, self._pdm, **ablkwargs)
        return self.ablCkpt
        
    def _calculateUpdates(
            self, stepper, epoch, iteration, totit, losses, datasetFileName,
            devdataset_filename, force=False):
        ganTrainer = GanTrainer(**self._kwargs)
        updates = {}
        if force or mt.shouldDo(self._iterctr, self._kwargs["logPer"]):
            # TODO losses is goofy
            logging.getLogger(__name__).info("Updating losses: %s" % losses)
            updates.update(losses)
        if force or mt.shouldDo(self._iterctr, self._kwargs["logDevPer"]):
            logging.getLogger(__name__).info("Getting devset losses")
            dataset = torch.load(devdataset_filename)
            updates.update(stepper.getLoss(dataset, "Dev "))
        if force or (iteration == (totit) and mt.shouldDo(epoch, self._kwargs["updateAblationPer"])):
            ablCkpt = self.getAblCkpt(datasetFileName)
            startEpoch = ablCkpt.epoch
            ablCkpt._kwargs["epochs"] = epoch
            logging.getLogger(__name__).info("Training ablckpt from %d to %d" %\
                    (ablCkpt.epoch, ablCkpt._kwargs["epochs"]))
            ablCkpt.train()

            abls = evGanAndAbl((stepper.netD, stepper.netG), (ablCkpt.netD, ablCkpt.netG), datasetFileName, self._pdm, ganTrainer, **self._kwargs)
            updates.update(abls)
            abls = evGanAndAbl(
                    (stepper.netD, stepper.netG), (ablCkpt.netD, ablCkpt.netG),
                    devdataset_filename, self._pdm, ganTrainer, **self._kwargs)
            for k,v in abls.iteritems():
                updates["Dev " + k] = v

        if force or (iteration == (totit) and mt.shouldDo(epoch, self._kwargs["updateParzenPer"])):
            probs = evaluateParzenProb(
                    stepper.netG, datasetFileName, self._pdm, **self._kwargs)
#def evaluateParzenProb(netG, ganDataSet, pairtraindir, featdir, **kwargs):
            updates.update(probs)
            probs = evaluateParzenProb(
                    stepper.netG, devdataset_filename, self._pdm,
                    **self._kwargs)
            for k,v in probs.iteritems():
                updates["Dev " + k] = v
        return updates
    def _appendValues(self, epoch, valdict):
        argdict = {k: pd.Series([v], index=[epoch]) for k,v in valdict.iteritems()}
        self._data = self._data.append(pd.DataFrame(argdict))
    def _makeGraphs(self):
        for gc in self._graphconfig:
            try:
                self._makeOneGraph(gc)
            except Exception as e:
                logging.getLogger(__name__).info("Error making graph: %s" % str(e))
        # Write the source-of-truth
        dest = self._graphbn + "source.csv"
        tmp = mt.pathinsert(dest, "tmp")
        self._data.to_csv(tmp)
        os.rename(tmp, dest)

    def _makeOneGraph(self, gc):
        chosen = gc["chosen"]
        mydata = self._data[chosen].dropna()
        x = mydata.index.values
        ySeriesIterable = mydata.values.T
        yLegendIterable = chosen
        temploc = mt.pathinsert(gc["loc"], "tmp")
        gc.get("maker", pu.makeNPlotDefault)(
                x, ySeriesIterable, yLegendIterable, gc["title"], gc["xlabel"],
                gc["ylabel"], temploc, **gc.get("kwargs", {}))
        os.rename(temploc, gc["loc"])

def trainCGANNoFile(
        datasetFileName, devDatasetFileName, pairDataManager, **kwargs):
  trainer = CheckpointTrainer(
          datasetFileName, devDatasetFileName, 
          pairDataManager, **kwargs)
  return trainer.train()

class CheckpointTrainer(object):
    def __init__(
            self, datasetFileName, devDatasetFileName, pairDataManager,
            **kwargs):
        self._kwargs = kwargs
        self.datasetFileName = datasetFileName
        self.devDatasetFileName = devDatasetFileName
        self.pairDataManager = pairDataManager
        self.epoch = 0

        mapBaseName = kwargs.get("mapBaseName", datasetFileName)
        role2Int = torch.load("%s_role2Int" % mapBaseName)
        noun2Int = torch.load("%s_noun2Int" % mapBaseName)
        self._nVRs = max(role2Int.values()) + 1
        self._nWords = max(noun2Int.values()) + 1
        self._wESize = 128
        self._vrESize = 128

        self._lr = kwargs["lr"] # This will change during training, potentially.
        netD, netG = self.makeNets()

        self.trainGraphGenerator = TrainGraphGenerator(
                self.pairDataManager, self._kwargs.get("trainGraphBn", None),
                **self._kwargs)

        self.updater = UpdateStepper(
                self.pairDataManager, netD, netG, **self._kwargs)

    @property
    def netD(self):
        return self.updater.netD
    @property
    def netG(self):
        return self.updater.netG
    def makeNets(self):
        style = self._kwargs.get("style", "trgan")
        if style == "gan":
            netG = NetG(
                    self._kwargs["nz"], pc.IMFEATS, self._kwargs["hiddenSize"],
                    self._kwargs["genDepth"], self._nWords, self._wESize,
                    self._nVRs, self._vrESize, self._kwargs["gdropout"]).cuda(
                            self._kwargs["gpu_id"])
            netD = NetD(
                    pc.IMFEATS, 1, self._kwargs["hiddenSize"],
                    self._kwargs["depth"], self._nWords, self._wESize,
                    self._nVRs, self._vrESize).cuda(self._kwargs["gpu_id"])
        elif style == "trgan":
            netG = TrGanG(
                    self._kwargs["nz"], pc.IMFEATS, self._kwargs["hiddenSize"],
                    self._kwargs["genDepth"], self._nWords, self._wESize,
                    self._nVRs, self._vrESize, self._kwargs["gdropout"]).cuda(
                            self._kwargs["gpu_id"])
            netD = TrGanD(
                    pc.IMFEATS, 1, self._kwargs["hiddenSize"],
                    self._kwargs["depth"], self._nWords, self._wESize,
                    self._nVRs, self._vrESize, self._kwargs["trgandnoImg"]) \
                        .cuda(self._kwargs["gpu_id"])
        else:
            raise ValueError("Invalid style '%s'" % style)
        return netD, netG

    def makeSavedir(self):
        if self._kwargs.get("saveDir", None) is not None:
          if os.path.exists(saveDir):
            logging.getLogger(__name__).info(
                "Save directory %s already exists; cowardly exiting..." % saveDir)
            sys.exit(1)
          os.makedirs(saveDir)

    def train(self):
        logging.getLogger(__name__).info(
                "Loading dataset from %s" % self.datasetFileName)
        dataset = torch.load(self.datasetFileName)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self._kwargs["batchSize"], shuffle=False, num_workers=0)
        self.makeSavedir()

        totit = len(dataloader)
        if self.epoch == 0:
            self.trainGraphGenerator.logInitialState(
                    self.datasetFileName, self.devDatasetFileName, totit, 
                    self.updater)
        for self.epoch in tqdm.tqdm(
                range(self.epoch, self._kwargs["epochs"]),
                total=self._kwargs["epochs"], desc="Epochs completed: ",
                disable=self._kwargs.get("disablecgantrainprog", True)):
            for i, data in tqdm.tqdm(
                    enumerate(dataloader, 0), total=len(dataloader),
                    desc="Iterations completed: ", leave=False,
                    disable=self._kwargs.get("disablecgantrainprog", True)):


                lossdict = self.updater.networkUpdateStep(data)

                # TODO we will want validation data for this as well.
                self.trainGraphGenerator.generate(
                        self.updater, self.epoch, i + 1,
                        totit, lossdict, self.datasetFileName, 
                        self.devDatasetFileName)
            # save the training files
            self.saveCheckpointsIfNeeded()

            self.adjustLearnRate()
        return self.netD, self.netG

    def adjustLearnRate(self):
        decayPer = self._kwargs.get("decayPer", None)
        if decayPer is None:
            return
        if mt.shouldDo(self.epoch, decayPer):
            self._lr *= self._kwargs["decayRate"]
            for param_group in self.updater._optimizerG.param_groups:
                logging.getLogger(__name__).info("optG: prev rate: %f" % param_group['lr'])
                param_group['lr'] = self._lr
                logging.getLogger(__name__).info("optG: new  rate: %f" % param_group['lr'])
            for param_group in self.updater._optimizerD.param_groups:
              logging.getLogger(__name__).info("optD: prev rate: %f" % param_group['lr'])
              param_group['lr'] = self._lr
              logging.getLogger(__name__).info("optD: new  rate: %f" % param_group['lr'])
    def saveCheckpointsIfNeeded(self):
        savePerEpoch = self._kwargs.get("savePerEpoch", 5)
        ganSaveBasename = self._kwargs.get("ganSaveBasename", None)
        if self._kwargs.get("saveDir", None) is not None and ganSaveBasename is not None and self._epoch % savePerEpoch == (savePerEpoch - 1):
            checkpointName = "%s_epoch%d.pyt" % (os.path.basename(ganSaveBasename), epoch)
            logging.getLogger(__name__).info("Saving checkpoint to %s" % checkpointName)
            torch.save((netD, netG), os.path.join(saveDir, checkpointName))
        
class UpdateStepper(object):
    REAL_LABEL = 1
    FAKE_LABEL = 0
    def __init__(self, pairDataManager, netD, netG, **kwargs):
        self._kwargs = kwargs
        self._pdm = pairDataManager
        self.netD = netD
        self.netG = netG

        # TODO update the learning rate here!

        self._noise = ag.Variable(torch.FloatTensor(self._kwargs["batchSize"], self._kwargs["nz"]).cuda(self._kwargs["gpu_id"]))
        self._label = ag.Variable(torch.FloatTensor(self._kwargs["batchSize"]).cuda(self._kwargs["gpu_id"]))
        self._optimizerD = optim.SGD(
                self.netD.parameters(), lr=self._kwargs["lr"])
        self._optimizerG = optim.SGD(
                self.netG.parameters(), lr=self._kwargs["lr"])

        self._criterion = self._setCriterion(self._kwargs["losstype"]).cuda(self._kwargs["gpu_id"])
        self._l1Criterion = nn.L1Loss().cuda(self._kwargs["gpu_id"])
    def _setCriterion(self, name):
        # TODO this bce will be sadtown if used.
        mapping = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}
        return mapping[name]

    def getLoss(self, dataset, key_prefix):
        # TODO look here; why isn't it going?
        # dataset - a TensorDataset
        # This dataset needs to be a regular-style dataset, not a gan-style one.
        values = {}
        batch_size = self._kwargs["batchSize"]
        tot_points = 0.
        dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, 
                num_workers=0)
        for i, data in tqdm.tqdm(
                enumerate(dataloader, 0), total=len(dataloader),
                desc="Iterations completed: ", leave=False,
                disable=self._kwargs.get("disablecgantrainprog", True)):
            # calculate D loss
            # TODO multiply by batch sizse!
            dloss, _ = self.calculateDLoss(data)
            gloss, _ = self.calculateGLoss(data)

            for k,v in it.chain(dloss.iteritems(), gloss.iteritems()):
                key = "%s%s" % (key_prefix, k)
                values[key] = values.get(key, 0.) + batch_size * v
            tot_points += batch_size

        for k in values.iterkeys():
            values[k] /= tot_points
        return values

    def calcCriterion(self, output, label, scores):
        loss = self._criterion(output.view(-1), label[:len(output)])
        if self._kwargs["useScore"]:
            loss = torch.mul(loss, scores)
        return loss
    def updateD(self, data):
        losses, errors = self.calculateDLoss(data)
        for error in errors:
            error.backward()
        self._optimizerD.step()
        return losses
    # Wait a minute, there are 2 terms in D-Loss... hence, it's bigger...?
    def calculateDLoss(self, data):
        self.netD.zero_grad()
        # TODO is this the most efficient way to get data onto the GPU?
        conditional, realImageAndScore = data
        conditional = ag.Variable(conditional.cuda(self._kwargs["gpu_id"]))
        realImageAndScore = ag.Variable(realImageAndScore.cuda(self._kwargs["gpu_id"]))
        _, scores = self._pdm.decodeFeatsAndScore(realImageAndScore)

        output = self.netD(conditional, realImageAndScore, train=True)
        self._label.data.fill_(self.REAL_LABEL)
        errD_real = self.calcCriterion(output, self._label, scores)
        # D_x is how many of the images (all of which were real) were identified
        # as real.
        D_x = output.data.mean()

        self._noise.data.normal_(0, 1)
        fake = self.netG(
                self._noise[:len(realImageAndScore)], conditional, train=True,
                ignoreCond=self._kwargs.get("ignoreCond", False))
        self._label.data.fill_(self.FAKE_LABEL)
        output = self.netD(conditional, fake, train=True)
        errD_fake = self.calcCriterion(output, self._label, scores)
        # D_G_z1 is how many of the images (all of which were fake) were 
        # identified as real.
        D_G_z1 = output.data.mean()
        errD = (errD_real + errD_fake) / 2.
        # TODO should I return a different value, or an average?
        return {
            "Loss_D": errD.data[0],
            "Loss_D_real": errD_real.data[0],
            "Loss_D_fake": errD_fake.data[0],
            "D_G_z1": D_G_z1,
            "D(x)": D_x}, [errD_real, errD_fake]
    def calcL1Loss(self, fakeIm, realIm, scores):
        l1loss = torch.mul(
                self._l1Criterion(fakeIm, realIm), self._kwargs["lam"])
        if self._kwargs["useScore"]:
            resizescores = scores.contiguous().view(-1, 1).expand(fake.size())
            l1loss = torch.mul(l1loss, resizescores)
        return l1loss
        
    def updateG(self, data):
        losses, errors = self.calculateGLoss(data)
        for error in errors:
            error.backward()
        self._optimizerG.step()
        return losses

    def calculateGLoss(self, data):
        # Update G network
        self.netG.zero_grad()
        conditional, realImageAndScore = data
        conditional = ag.Variable(conditional.cuda(self._kwargs["gpu_id"]))
        realImageAndScore = ag.Variable(realImageAndScore.cuda(self._kwargs["gpu_id"]))
        realIm, scores = self._pdm.decodeFeatsAndScore(realImageAndScore)

        self._noise.data.normal_(0, 1)
        fake = self.netG(
                self._noise[:len(realImageAndScore)], conditional, train=True,
                ignoreCond=self._kwargs.get("ignoreCond"))
        self._label.data.fill_(self.REAL_LABEL)
        output = self.netD(conditional, fake, train=True)
        errG = self.calcCriterion(output, self._label, scores)
        # D_G_z2 is how many of the images (all of which were fake) were
        # identified as real, AFTER the discriminator update.
        D_G_z2 = output.data.mean()

        fake = self.netG(
                self._noise[:len(realImageAndScore)], conditional, train=True,
                ignoreCond=self._kwargs.get("ignoreCond"))
        errG_L1 = self.calcL1Loss(fake, realIm, scores)
        self._optimizerG.step()
        return {
                "Loss_G": errG.data[0],
                "Loss_L1": errG_L1.data[0],
                "D_G_z2": D_G_z2}, [errG, errG_L1]
        
    def networkUpdateStep(self, data):
      ret = {}
      # Update D network
      for _ in range(self._kwargs["dUpdates"]):
        ret.update(self.updateD(data))
      for _ in range(self._kwargs["gUpdates"]):
        ret.update(self.updateG(data))
      return ret

def evaluateGANModelAndAblation(ganModelFile, *args, **kwargs):
  nets = torch.load(ganModelFile)
  evaluateGANModelAndAblationNoFile(nets, *args, **kwargs)

def evGanAndAbl(
        nets, ablnets, datasetFileName, pairDataManager, ganTrainer,
        **kwargs):
    logging.getLogger(__name__).info("Evaluating original model")
    trueLoss = evaluateGANModelNoFile(nets, datasetFileName, **kwargs)
    logging.getLogger(__name__).info("Evaluating ablation model")
    ablatedLoss = evaluateGANModelNoFile(ablnets, datasetFileName, **kwargs)
    return {"True Loss": trueLoss, "Ablated Loss": ablatedLoss}

def evaluateGANModelAndAblationNoFile(nets, datasetFileName, pairDataManager, ganTrainer, epochOverride, **kwargs):
  logging.getLogger(__name__).info("Training ablation")
  kwargs["ganSaveBasename"] = None
  kwargs["ignoreCond"] = True
  kwargs["trainGraphBn"] = None
  kwargs["epochs"] = epochOverride
  ablnets = ganTrainer.trainNoFile(datasetFileName, pairDataManager, **kwargs)
  return evGanAndAbl(
      nets, ablnets, datasetFileName, pairDataManager, ganTrainer, **kwargs)

def evaluateGANModel(ganModelFile, datasetFileName, **kwargs):
  nets = torch.load(ganModelFile)
  return evaluateGANModelNoFile(nets, datasetFileName, **kwargs)

def evaluateGANModelNoFile(nets, datasetFileName, **kwargs):
  logging.getLogger(__name__).info("Evaluating gan model: kwargs=%s" % str(kwargs))
  gpu_id = kwargs["gpu_id"]
  useScore = kwargs["useScore"]
  disableevganprog = kwargs.get("disableevganprog", True)
  logging.getLogger(__name__).info("Evaluating gan model on gpu %d" % gpu_id)
  batchSize=16
  """
  Evaluate a GAN model generator's image creation ability in a simple way.
  """
  netD, netG = nets

  nz = netG.inputSize

  netG = netG.cuda(gpu_id)
  netD = netD.cuda(gpu_id)
  dataset = torch.load(datasetFileName)
  # This dataset needs to be a regular-style dataset, not a gan-style one.
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batchSize, shuffle=True, num_workers=0)
  criterion = nn.MSELoss()
  runningTrue = 0.
  noise = ag.Variable(torch.FloatTensor(dataloader.batch_size, nz).cuda(gpu_id))
  datasetSize = 0.
  for i, data in tqdm.tqdm(
        enumerate(dataloader, 0), total=len(dataloader),
        desc="Iterations completed: ", disable=disableevganprog):
    expectedDataSize = 12 + 3 + pc.IMFEATS
    assert(len(data[0][0]) == expectedDataSize), \
        "expected len(data[0][0])=%d to be %d" % \
        (len(data[0][0]), expectedDataSize)
    expectedYDataSize = pc.IMFEATS + 1
    assert(len(data[1][0]) == expectedYDataSize), \
        "expected len(data[0][0])=%d to be %d" % \
        (len(data[1][0]), expectedYDataSize)
    # 1. No labels on generator. Calculate square loss.
    conditionals, imageAndScore = data
    conditionals = conditionals.cuda(gpu_id)
    scores = imageAndScore[:,-1].cuda(gpu_id).contiguous().view(-1, 1)
    if not useScore:
      scores.fill_(1)
    noise.data.normal_(0, 1)
    batchSize = len(conditionals)
    output = netG(
        noise[:batchSize], ag.Variable(conditionals.cuda(gpu_id),
            volatile=True), True if nz < 1 else False, ignoreCond=False)
    expectedFeatures = imageAndScore[:,:pc.IMFEATS].cuda(gpu_id).contiguous()
    expandedScores = scores.expand(output.size())

    correctLoss = criterion(
        ag.Variable(torch.mul(output.data, expandedScores), requires_grad=False), 
        ag.Variable(torch.mul(expectedFeatures, expandedScores), requires_grad=False))

    datasetSize += batchSize

    runningTrue += correctLoss.data[0] * batchSize
  runningTrue /= datasetSize
  logging.getLogger(__name__).info("True   Loss: %.4f" % runningTrue)
  return runningTrue

class GanDataLoader(object):
  def __init__(self, filename):
    self.filename = filename
    self.data = torch.load(self.filename)
    self.validate()
  def validate(self):
    if len(self.data) == 0:
      raise Exception("GanDataLoader", "No data found")
    x, y = self.data[0]
    assert(len(x) == 12 + 3 + pc.IMFEATS), "Invalid 'x' dimension '%d'" % len(x)
    assert(len(y) == pc.IMFEATS + 1), "Invalid 'y' dimension '%d'" % len(y)

def getGANChimeras(netG, nTestPoints, yval, dropoutOn=True, **kwargs):
  gpu_id = kwargs["gpu_id"]
  batchSize = kwargs["batchSize"]
  out = []
  nz = netG.inputSize
  noise = ag.Variable(
      torch.FloatTensor(batchSize, nz), requires_grad=False).cuda(gpu_id)
  yvar = ag.Variable(yval.expand(batchSize, yval.size(1))).cuda(gpu_id)
  for i in range(nTestPoints):
    noise.data.normal_(0, 1)
    if i * batchSize > nTestPoints:
      break
    out = torch.cat(out + [netG(noise, yvar, dropoutOn, ignoreCond=False)], 0)
    out = [out]
  return out[0][:nTestPoints]

def parzenWindowFromFile(ganModelFile, *args, **kwargs):
  _, netG = torch.load(ganModelFile)
  return parzenWindowGanAndDevFile(netG, *args, **kwargs)

def parzenWindowGanAndDevFile(netG, ganDevSet, pairDataManager, **kwargs):
  gDiskDevLoader = GanDataLoader(ganDevSet)
  return parzenWindowProb(netG, gDiskDevLoader.data, pairDataManager, **kwargs)

def evaluateParzenProb(netG, datasetFileName, pairDataManager, **kwargs):
    try:
      probs = parzenWindowGanAndDevFile(netG, datasetFileName, pairDataManager, **kwargs)
    except Exception as e:
      logging.getLogger(__name__).info("Error computing parzen value: %s" % str(e))
      # TODO
      return {"Parzen Average": 0}
    # TODO just take a flat average of them, I guess?
    # TODO this method should also give back dev set stuff.
    _sum = 0.
    _den = 0.
    for k,v in probs.iteritems():
        _sum += np.sum(v)
        _den += len(v)
    return {"Parzen Average": _sum / _den}

def parzenWindowProb(netG, devData, pairDataManager, **kwargs):
  # TODO this is using ydata as the conditional! It shouldn't.
  # I need to fix many things about this function.
  # TODO with only 5k datapoints, this runs a 12G gpu out of memory. Seems
  # like it shouldn't... I should 
  logging.getLogger(__name__).info("Making parzen window with settings %s" % \
      str(kwargs))
  """
  nSamples - number of points to draw from generator when making distribution
  nTestSamples - number of points to draw from devData for testing
  """
  # Set up some values.
  gpu_id = kwargs["gpu_id"]
  logPer = kwargs.get("logPer", 1e12)
  bw_method = kwargs.get("bw_method", "scott")
  nSamples = kwargs.get("nSamples", 5000)
  nTestSamples = kwargs.get("nTestSamples", 100)
  style = kwargs.get("style", "trgan")
  test = kwargs.get("test", False)
  disableparzprog = kwargs.get("disableparzprog", True)

  logging.getLogger(__name__).info("Running parzen fit on gpu %d" % gpu_id)

  netG = netG.cuda(gpu_id)

  # map from conditioned value to a list of datapoints that match that value.
  ymap = collections.defaultdict(list)

  # This batch_size must be one, because I don't iterate over ydata again.
  devDataLoader = torch.utils.data.DataLoader(
      devData, batch_size=1, shuffle=False, num_workers=0)
  logging.getLogger(__name__).info("Testing dev data against parzen window fit")
  for i, data in tqdm.tqdm(
      enumerate(devDataLoader, 0), total=len(devDataLoader),
      desc="Iterations completed: ", disable=disableparzprog):
    if i >= nTestSamples:
      break
    fullConditional, imageAndScore = data
    image, _ = pairDataManager.decodeFeatsAndScore(imageAndScore)
    conditional = pairDataManager.getConditionalStyle(fullConditional, style)
    # kde_input is pc.IMFEATSx1
    kde_input = torch.transpose(image, 0, 1).numpy()
    key = tuple(*conditional.numpy().tolist())
    ymap[key].append((kde_input, fullConditional))

  probs = {}
  for i, (relevantCond, kdeinAndFC) in tqdm.tqdm(
      enumerate(ymap.iteritems()), total=len(ymap), disable=disableparzprog):
    kdein, fullConditional = zip(*kdeinAndFC)

    # Convert conditional back into a tensor. By definition, each element should
    # be the same (or in style == 'gan' case, the elements that aren't the same
    # are irrelevant.
    fullConditional = fullConditional[0]
    assert(fullConditional.size() == torch.Size([1, pc.FCSIZE])), \
        "Invalid fullConditional.size()=%s" % str(fullConditional.size())


    dataTensor = getGANChimeras(
        netG, nSamples, fullConditional, 
        dropoutOn=True if netG.inputSize < 1 else False, **kwargs)
    assert(dataTensor.size() == torch.Size([nSamples, pc.IMFEATS])), \
        "Invalid dataTensor.size()=%s" % str(dataTensor.size())
    kde_train_input = torch.transpose(dataTensor, 0, 1).data.cpu().numpy()
    gpw = kde.gaussian_kde(kde_train_input, bw_method=bw_method)

    devData = np.concatenate(kdein, axis=1)
    probs[relevantCond] = gpw.logpdf(devData)
    if i % logPer == (logPer - 1):
      logging.getLogger(__name__).info("Investigating conditional %s" % pairDataManager.condToString(fullConditional))
      logging.getLogger(__name__).info("number of x points: %d" % len(kdein))
      logging.getLogger(__name__).info(
          "prob sample: %s" % repr(probs[relevantCond]))
    if test:
      return probs

  return probs
