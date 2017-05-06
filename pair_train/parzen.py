import data as dataman
import hack.stats.kde as kde
import numpy as np
import gan as gan
import pair_train.nn as nn
import utils.mylogger as logging
import torch
import pair_train.constants as pc
import collections
import tqdm

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

def try_parzen():
    ganfile = "data/smalltest3/multigan/trained_models/1124_1332.gan"
    gantrainset = "data/smalltest3/multigan/nounpair_data/1124_1332.data"
    # TODO cross-validate?
    for factor in xrange(0, 25):
        method = 2 ** factor
        #method = "scott"
        pdm = dataman.PairDataManager("data/smalltest3/data/pairs/", "data/regression_fc7/")
        probs = gan.parzenWindowFromFile(ganfile, gantrainset, pdm, logPer=100, bw_method=method, test=False, gpu_id=0, batchSize=64)
        logging.getLogger(__name__).info("base=%d, probs=%s" % (factor, str(probs.values())))

def parzenWindowFromFile(ganModelFile, *args, **kwargs):
  _, netG = torch.load(ganModelFile)
  return parzenWindowGanAndDevFile(netG, *args, **kwargs)

def parzenWindowGanAndDevFile(netG, ganDevSet, pairDataManager, **kwargs):
  gDiskDevLoader = GanDataLoader(ganDevSet)
  return parzenWindowProb(netG, gDiskDevLoader.data, pairDataManager, **kwargs)

def evaluateParzenProb(netG, datasetFileName, pairDataManager, **kwargs):
    probs = parzenWindowGanAndDevFile(netG, datasetFileName, pairDataManager, **kwargs)
    _sum = 0.
    _den = 0.
    for k,v in probs.iteritems():
        _sum += np.sum(v)
        _den += len(v)
    return {"Parzen Average": _sum / _den}

def parzenWindowProb(netG, devData, pairDataManager, **kwargs):
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
    # TODO this key business probably isn't needed.

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


    dataTensor = dataman.getGANChimeras(
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

if __name__ == '__main__':
    try_parzen()
