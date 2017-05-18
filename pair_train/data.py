import json
import torch.autograd as ag
import utils.mylogger as logging
import os
import re
import torch
import utils.methods as mt
import numpy as np
import constants as pc # short for pair_train constants
import copy
import split.data_utils as du
import collections
import torch.utils.data as td
import tqdm
import sys

def getNSamplesFromDatafile(filename):
    return len(torch.load(filename))

class ShardedDataHandler(object):
  # Class for reading the folder output of shardAndSave
  def __init__(self, directory, suffix = ".data"):
    self.directory = directory
    self._suffix = suffix
    if not os.path.exists(self.directory):
      os.makedirs(self.directory)
  def keyToName(self, key):
    return "_".join(map(lambda x: str(int(x)), key)) + self._suffix
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
  def nameToNouns(self, filename):
    relevant = os.path.basename(filename)
    fileReg = re.compile("(\d+)(?:.0)?_(\d+)(?:.0)?" + self._suffix)
    match = fileReg.match(relevant)
    if not match:
      raise ValueError("Invalid filename '%s' doesn't match regex")
    return map(lambda x: int(float(x)), match.group(1, 2))
  def iterNounPairs(self):
    for filename in os.listdir(self.directory):
      yield self.nameToNouns(filename)
  def keyToPath(self, key):
    return os.path.join(self.directory, self.keyToName(key))

def shardAndSave(pdm, outDirName, dataset, **kwargs):
  """
  Break up data in datasetFileName based on noun pairs. Save the chunks of data
  to folder outDirName.
  """
  if dataset == "train":
    datasetFileName = pdm.trainLoc
  elif dataset == "dev":
    datasetFileName = pdm.devLoc
  else:
    raise ValueError("Invalid dataset '%s'" % str(dataset))

  logging.getLogger(__name__).info("Sharding data in %s" % datasetFileName)

  minDataPts = kwargs.get("minDataPts", 0)
  nounpairToData = collections.defaultdict(list)
  dataSet = torch.load(datasetFileName)
  dataloader = td.DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=0)
  logging.getLogger(__name__).info("Found %d datapoints" % len(dataloader))
  for i, data in tqdm.tqdm(
      enumerate(dataloader, 0), total=len(dataloader),
      desc="Sharding data...", leave=False):
    conditionals, testImageAndScore = data

    _, _, n1, n2, _ = pdm.decodeCond(conditionals)

    key = (n1[0,], n2[0,]) # Get n1, n2
    nounpairToData[key].append((data[0].clone(), data[1].clone()))
  logging.getLogger(__name__).info(
      "Got %d unique noun pairs" % len(nounpairToData))
  bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 10000, sys.maxint]
  distribution = np.histogram(
      [len(v) for v in nounpairToData.itervalues()], bins=bins)
  logging.getLogger(__name__).info(
      "Distribution of dataset sizes: %s" % mt.histString(distribution))
  logging.getLogger(__name__).info(
      "Requiring at least %d datapoints" % minDataPts)
  nounpairToData = {
      pair: data for pair, data in nounpairToData.iteritems() if len(data) > minDataPts}
  logging.getLogger(__name__).info(
      "Filtered down to %d unique noun pairs" % len(nounpairToData))

  train_only = kwargs.get("train_only", None)
  if train_only:
    new_data = {}
    for key in train_only:
        semantic_key = (pdm.noun2int[key[0]], pdm.noun2int[key[1]])
        new_data[semantic_key] = nounpairToData[semantic_key]
    nounpairToData = new_data
    logging.getLogger(__name__).info("Used train_only to filter to %d points" %
        len(nounpairToData))
    
  saver = ShardedDataHandler(outDirName)
  saver.save(nounpairToData)

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

class PairDataManager(object):
    # TODO rely on this class for data reading / investigating. Remove
    # hard-coded assumptions on layout of data
    def __init__(self, directory, featDir):
        self._directory = directory
        self._andecoder = os.path
        self.trainLoc = os.path.join(self._directory, "pairtrain.pyt")
        self.devLoc = os.path.join(self._directory, "pairdev.pyt")
        self._role2intFile = "%s_role2Int" % self.trainLoc
        self._noun2intFile = "%s_noun2Int" % self.trainLoc
        self._verb2intFile = "%s_verb2Int" % self.trainLoc
        self._featDir = featDir

        self.train_img_names = set(json.load(
                open("_".join([self.trainLoc, "trainImgNames.json"]))))
        self.dev_img_names = set(json.load(
                open("_".join([self.devLoc, "devImgNames.json"]))))

        self.role2int = torch.load(self._role2intFile)
        self.noun2int = torch.load(self._noun2intFile)
        self.verb2int = torch.load(self._verb2intFile)

        self.int2role = mt.reverseMap(self.role2int)
        self.int2noun = mt.reverseMap(self.noun2int)
        self.int2verb = mt.reverseMap(self.verb2int)

        imnames = os.listdir(self._featDir)
        self.name2feat = getIm2Feats(self._featDir, imnames)
        self._feat2name = { tuple(v): k for k,v in self.name2feat.iteritems()}

    def getImnameFromFeats(self, feats):
        return self._feat2name[tuple(np.array(feats.view(pc.IMFEATS).numpy(), dtype=np.float64))]   

    def noun2String(self, noun):
        return du.decodeNoun(self.int2noun[noun])

    def partdecodeToFulldecode(self, partlyDecodedAn):
        ret = copy.copy(partlyDecodedAn)
        for i, samp in enumerate(partlyDecodedAn):
            if i % 2 == 1:
                ret[i] = du.decodeNoun(samp)
        return ret
    def decodeCond(self, cond):
        assert(cond.size(1) == 1039), "Invalid cond, size=%d" % cond.size(1)
        return cond[:,:12], cond[:,12], cond[:,13], cond[:,14], cond[:,15:]
    def decodeFeatsAndScore(self, fas):
        assert(fas.size(1) == pc.IMFEATS + 1), "Invalid feats-and-score!"
        return fas[:,:pc.IMFEATS], fas[:,pc.IMFEATS]
    def getConditionalStyle(self, conditional, style):
        if style == "gan":
            return conditional[:15]
        elif style == "trgan":
            return conditional
        raise ValueError("Invalid style '%s'" % style)
    def condToString(self, cond):
        annotations, role, n1, n2, sourceFeats = self.decodeCond(cond)

        partlyDecodedAn = self.annotationsintsToCodes(annotations)
        fullyDecodedAn = self.partdecodeToFulldecode(partlyDecodedAn)
        roleStr = str(self.int2role.get(role[0,], "MISSING"))
        def nounstr(noun):
            return str(du.decodeNoun(self.int2noun[noun]))
        n1Str = nounstr(n1[0,])
        n2Str = nounstr(n2[0,])
        return " | ".join(map(str, (fullyDecodedAn, roleStr, n1Str, n2Str)))
    def get_cond_from_td(self, tensor_dataset):
        return torch.cat(zip(*tensor_dataset)[1])
    # TODO what to do about values that are too big?
    def annotationsintsToCodes(self, annotations):
        assert(annotations.size() == torch.Size((1, 12))), "Invalid annotations"
        anList = list(annotations.view(12))

        ret = []
        for i, elem in enumerate(anList):
            if i % 2 == 0:
                ret.append(self.int2role.get(elem, ""))
            else:
                ret.append(self.int2noun.get(elem, ""))
        return ret

def getIm2Feats(featureDirectory, imNames):
  imToFeatures = {name: np.fromfile("%s%s" % (featureDirectory, name), dtype=np.float32) for name in imNames } # Should have all names in it.
  imToFeatures = {name: np.array(values, dtype=np.float64) for name, values in imToFeatures.iteritems() }
  return imToFeatures

def getGANChimeras(netG, nTestPoints, yval, dropoutOn=True, **kwargs):
  """
  @param yval - a 1xD torch.Tensor with the conditional values to be generated.
  """
  gpu_id = kwargs["gpu_id"]
  yval = yval.expand(nTestPoints, yval.size(1)).cuda(gpu_id)
  return getGANChimerasForYvals(netG, yval, dropoutOn, **kwargs)[:nTestPoints]

def getGANChimerasForYvals(netG, yvals, dropoutOn=True, **kwargs):
  gpu_id = kwargs["gpu_id"]
  out = []
  nz = netG.inputSize
  noise = ag.Variable(
      torch.FloatTensor(yvals.size(0), nz), requires_grad=False).cuda(gpu_id)
  yvar = ag.Variable(yvals).cuda(gpu_id)
  return netG(noise, yvar, dropoutOn, ignoreCond=False)

def cond_by_helper(pdm, full_conditional):
    # Only want to condition by target semantics
    an, role, n1, n2, im_and_score = pdm.decodeCond(full_conditional)
    target_an = an.clone()
    # Replace the role-n1 pair with role-n2
    role = role[(0,)]
    n1 = n1[(0,)]
    n2 = n2[(0,)]
    for i in range(0, target_an.size(1), 2):
        if target_an[(0, i)] == role and target_an[(0, i + 1)] == n1:
            target_an[(0, i + 1)] = n2
        return target_an
    raise ValueError(
            "Invalid conditional: an=%s, role=%s, n1=%s, n2=%s, "
            "im_and_score=%s" % (an, role, n1, n2, im_and_score))

class CondSharder(object):
    """
    Capable of sharding a dataset into pieces that have the same conditional.
    """
    def __init__(self, style):
        self._style = style
    def shard(self, condition_by, dataset):
        """
        condition_by: something called on full_conditional, to get the key
        dataset - a torch.TensorDataset containing the data.
        """
        results = collections.defaultdict(list)
        dataloader = td.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for i, data in enumerate(dataloader, 0):
            full_conditional, im_and_score = data
            conditional = condition_by(full_conditional)
            results[conditional].append(data)

        return results
