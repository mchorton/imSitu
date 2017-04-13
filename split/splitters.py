import data_utils as du
import os
import utils.mylogger as logging
def splitTrainDevTestMinInTrain(outDir, test=False):
  # set random seed
  import random
  random.seed(331900)

  if not os.path.exists(outDir):
    os.makedirs(outDir)
  """
  randevsize = 1
  rantestsize = 1
  devzs = 1
  testzs = 1
  datasets = ["testfunc.json"]
  """
  randevsize = 12000
  rantestsize = 12000
  devzs = 12000
  testzs = 12000
  datasets = ["train.json", "dev.json", "test.json"]

  finalSplit = {}
  logging.getLogger(__name__).info("Loading Data")
  full_data = du.get_joint_set(datasets)
  if test:
    npts = 1000
    randevsize = npts / 10
    rantestsize = npts / 10
    devzs = npts / 10
    testzs = npts / 10

    chosen = {}
    for i, (k,v) in enumerate(full_data.iteritems()):
      chosen[k] = v
      if i > npts:
        full_data = chosen
        break
  logging.getLogger(__name__).info("Getting image deps")
  imgdeps = du.getImageDeps(du.getvrn2Imgs(full_data))

  logging.getLogger(__name__).info("Getting uniform split")
  init = set(full_data.keys())
  init, finalSplit["zsRanDev"] = du.get_uniform_split(init, randevsize)
  init, finalSplit["zsRanTest"] = du.get_uniform_split(init, rantestsize)

  N = len(init)
  #softTrainLim = (N - (devzs + testzs) * 2) / (1. * N) # corresponds to 52%
  #hardTrainLim = (N - (devzs + testzs) * 2) / (1. * N)
  softTrainLim = 0.5
  hardTrainLim = 0.5

  finalSplit["zsTrain"], zsDevTestWaste = du.get_perverb_zssplit(init, softTrainLim, hardTrainLim, imgdeps)
  zsDevTest, finalSplit["waste"] = du.filterZeroShot(zsDevTestWaste, finalSplit["zsTrain"], imgdeps)
  finalSplit["zsDev"], finalSplit["zsTest"] = du.get_uniform_split(zsDevTest, len(zsDevTest) / 2)

  for k,v in finalSplit.iteritems():
    logging.getLogger(__name__).info(
        "Data set %s has %d points" % (str(k), len(v)))

  du.saveDatasets(finalSplit, full_data, outDir)
