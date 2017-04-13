import data_utils as du
# TODO This doesn't work, and should be fixed. Punting for now.
# Main issue is, I'm not sure where get_perverb_zssplit() went. Maybe I
# accidentally deleted it? TODO dig through git logs.
def splitTrainDevTestMinInTrain():
  # TODO automatically chosen!
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
  print "Loading Data"
  full_data = du.get_joint_set(datasets)
  imgdeps = du.getImageDeps(du.getvrn2Imgs(full_data))

  print "getting uniform split"
  init = set(full_data.keys())
  init, finalSplit["zsRanDev"] = du.get_uniform_split(init, randevsize)
  init, finalSplit["zsRanTest"] = du.get_uniform_split(init, rantestsize)

  N = len(init)
  #softTrainLim = (N - (devzs + testzs) * 2) / (1. * N) # corresponds to 52%
  #hardTrainLim = (N - (devzs + testzs) * 2) / (1. * N)
  softTrainLim = 0.5
  hardTrainLim = 0.5
  print N
  print softTrainLim
  print hardTrainLim
  #return N, int((devzs + testzs) * 1.1 * N)

  finalSplit["zsTrain"], zsDevTestWaste = get_perverb_zssplit(init, softTrainLim, hardTrainLim, imgdeps)
  zsDevTest, finalSplit["waste"] = du.filterZeroShot(zsDevTestWaste, finalSplit["zsTrain"], imgdeps)
  finalSplit["zsDev"], finalSplit["zsTest"] = du.get_uniform_split(zsDevTest, len(zsDevTest) / 2)

  for k,v in finalSplit.iteritems():
    print "split %s: %s" % (str(k), str(len(v)))

  saveDatasets(finalSplit, full_data)
