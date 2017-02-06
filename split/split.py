# 1: get dependency structure.
# 2: get 

# Note: when you move one image, you need to check how many images (in B)
# became 0-shot. You also need to consider their desired pairings.

import json
import copy
import collections
import random

MAXTHRESH=10

class ImgDep():
  def __init__(self, name):
    self.name = name
    # Map from verb-role to set of image names.
    self.deps = {} # TODO add_deps function?
    self.role2Nouns = {}
  def __str__(self):
    return "(%s, %s)" % (str(self.name), str(self.deps))
  def __repr__(self):
    return self.__str__()

  def isZS(self, trainImgDeps):
    """ Check whether this image is 0-shot """
    """
    for vr, imgs in self.deps:
      if all (key not in trainImgDeps for key in imgs):
        return True
    return False
    """
    return len(self.minNeededForZS(trainImgDeps)) == 0

  @property
  def verb(self):
    """
    Get the name of the verb represented in this image.
    """
    for k in self.deps:
      return k[0]
    return None

  def minNeededForZS(self, trainImgDeps):
    """ Get a list of how many movements are needed for 0-shot. """
    minset = set()
    minlen = 10**10
    for vr, imgs in self.deps.iteritems():
      #print "vr: %s" % str(vr)
      #print "imgs: %s" % str(imgs)
      curset = set([key for key in imgs if key in trainImgDeps])
      if len(curset) < minlen:
        minset = curset
        minlen = len(curset)
    return minset

  def numDifferentLabelings(self, otherImgDep):
    if self.verb is None or self.verb != otherImgDep.verb:
      return None
    numDifferences = 0
    for k,v in self.role2Nouns.iteritems():
      if len(otherImgDep.role2Nouns[k] & v) == 0:
        numDifferences += 1
    return numDifferences

def getvrn2Imgs(dataset):
  """
  Gets the (verb,role,noun) -> (set of images that use that combination), 
  from the original dataset.
  """
  vrn2Imgs = {}
  for imgname, labeling in dataset.iteritems():
    verb = labeling["verb"]
    for frame in labeling["frames"]:
      for role, noun in frame.iteritems():
        key = (verb, role, noun)
        vrn2Imgs[key] = vrn2Imgs.get(key, set()) | set([imgname])
  return vrn2Imgs

def getv2r(dataset):
  return {v["verb"]: set(v["frames"][0].keys()) for k,v in dataset.iteritems()}

def getImageDeps(vrn2Imgs):
  """
  Gets the ImgDep objects from a vrn2Imgs
  """
  imgdeps = {} # map from image name to ImgDep objects
  for vrn, imgs in vrn2Imgs.iteritems(): # TODO does this properly handle empty sets?
    for img in imgs:
      if img not in imgdeps:
        imgdeps[img] = ImgDep(img)
      vr = vrn[:2]
      imgdeps[img].deps[vr] = imgdeps[img].deps.get(vr, set()) | imgs
      imgdeps[img].role2Nouns[vr] = imgdeps[img].role2Nouns.get(vr, set()) | set([vrn[2]])
  return imgdeps

def get_joint_set(datasets):
  if  isinstance(datasets, str):
    datasets = [datasets]
  train = {}
  for dataset in datasets:
    train.update(json.load(open(dataset)))
  return train

def get_split(datasets, devsize):
  """ 
  Gets the desired splits of the data.
  @param datasets a list of datasets from which data will be merged, or a single filename
  @param devsize the % of data to put in the dev set
  """
  train = get_joint_set(datasets)
  vrn2Imgs = getvrn2Imgs(train)

  # get image dependencies.
  imgdeps = getImageDeps(vrn2Imgs)

  # Now, loop over the imgdeps, slowly adding things to set 'dev'
  # TODO could just be a keyset
  #trainDeps = copy.deepcopy(imgdeps) # Can just copy references. oh well. TODO
  trainDeps = set([k for k in imgdeps])
  devDeps = set()

  nDev = len(trainDeps) * devsize

  movethresh = 1
  cont = True
  while cont:
    print "train,dev,movethresh = %d,%d,%d" % (len(trainDeps), len(devDeps),movethresh)
    anyMoved = False
    for name,deps in imgdeps.iteritems():
      considerMoving = deps.minNeededForZS(trainDeps)
      #print "For image %s: considered moving %s" % (name, str(considerMoving))
      if len(considerMoving) <= movethresh and len(considerMoving) > 0:
        # Move all from train to dev
        trainDeps -= considerMoving
        devDeps |= considerMoving
        anyMoved = True
        if len(devDeps) > nDev:
          cont = False
          break
    if not anyMoved:
      movethresh += 1

  return trainDeps, devDeps, imgdeps

# TODO I should just modify the other one, to do what I expect...?
#  for img in devDeps:
#    if imgdeps[img].isZS(trainDeps):
#      nZero += 1
# It was a list, REKT!
def countZS(desiredSet, trainDeps, imgdeps):
  count = 0
  nZero = 0
  for img in desiredSet:
    count += 1
    if imgdeps[img].isZS(trainDeps):
      nZero += 1
  # TODO this is really slow! Why?
  return nZero
  #return len([img for img in desiredSet if imgdeps[img].isZS(trainDeps)])

def get_split_2(train, desiredZS=12000, ranDevSize=12000):
  """ 
  Gets the desired splits of the data.
  @param train the original training data, as a dictionary
  @param devsize the % of data to put in the dev set
  """
  # copy the train data; we intend to mangle it.
  cp = copy.deepcopy(train)
  train = cp

  ret = {}

  # Remove some data at random.
  ranDev = random.sample(train.keys(), ranDevSize)
  for key in ranDev:
    train.pop(key)
  ret["zsRanDev"] = set(ranDev)

  vrn2Imgs = getvrn2Imgs(train)

  # get image dependencies.
  imgdeps = getImageDeps(vrn2Imgs)

  # Now, loop over the imgdeps, slowly adding things to set 'dev'
  # TODO could just be a keyset
  #trainDeps = copy.deepcopy(imgdeps) # Can just copy references. oh well. TODO
  trainDeps = set([k for k in imgdeps]) # imgdeps.keys()
  devDeps = set()

  curZS = 0

  movethresh = 1
  cont = True
  while cont:
    print "train,dev,movethresh = %d,%d,%d" % (len(trainDeps), len(devDeps),movethresh)
    anyMoved = False
    for name,deps in imgdeps.iteritems():
      considerMoving = deps.minNeededForZS(trainDeps)
      if len(considerMoving) <= movethresh and len(considerMoving) > 0:
        # Move all from train to dev
        trainDeps -= considerMoving
        devDeps |= considerMoving
        anyMoved = True
    # count the number of 0-shot images
    nZS = countZS(devDeps, trainDeps, imgdeps)
    print "after iteration: obtained %s 0-shot, desire %s" % (str(nZS), desiredZS)
    if nZS >= desiredZS or len(trainDeps) == 0:
      cont = False
      break
    if not anyMoved:
      movethresh += 1

  ret["zsTrain"] = trainDeps
  ret["zsDev"] = devDeps
  return ret, imgdeps

# TODO test this.
def get_uniform_split(init, desiredOtherCount):
  """
  Assumes 'init' is a set.
  """
  ranSelection = set(random.sample(init, desiredOtherCount))
  newInit = set([k for k in init if k not in ranSelection])
  return newInit, ranSelection

# TODO should be able to rewrite old func. in terms of this.
# TODO want to know how many are 0-shot.
def moveSome(init, softlim, hardlim, imgdeps):
  print "moving some. softlim=%s, hardlim=%s" % (str(softlim), str(hardlim))
  orig = init
  moved = set()

  cont = True
  moveThresh = 1
  while cont:
    toMove = set()
    for img in orig:
      if len(toMove) + len(moved) > softlim:
        break
      considerMoving = imgdeps[img].minNeededForZS(orig)
      # TODO enforce hardcap. Should I also check softlim here?
      if len(considerMoving) <= moveThresh and len(considerMoving) > 0 and len(considerMoving) + len(moved) + len(toMove) < hardlim:
        toMove |= considerMoving
    orig -= toMove
    moved |= toMove
    if len(moved) >= softlim or len(moved) + moveThresh >= hardlim:
      break
    if len(toMove) == 0:
      moveThresh += 1
  return orig, moved

# TODO test this.
def get_perverb_zssplit(init, softTrainMin, hardTrainMin, imgdeps):
  """
  These limits are, amounts that must be in the training set!
  """
  def get_numerical_lim(lim, n):
    if lim < 1.:
      return n - int(lim * n)
    return n - lim

  # TODO make it so that softlim / hardlim is sometimes a count, sometimes a percent
  # Get a per-verb sharding of the initial data.
  sharding = collections.defaultdict(set)
  for img in init:
    sharding[imgdeps[img].verb].add(img)

  # Move through each shard. Move verbs until enough have been moved.
  cont = True
  orig = set()
  moved = set()
  for verb, imgs in sharding.iteritems():
    softlim = get_numerical_lim(softTrainMin, len(imgs))
    hardlim = get_numerical_lim(hardTrainMin, len(imgs))
    #print "verb=%s, softlim=%s, hardlim=%s" % (verb, str(softlim), str(hardlim))
    origcontrib, movedcontrib = moveSome(imgs, softlim, hardlim, imgdeps)
    orig |= origcontrib
    moved |= movedcontrib
    print "verb=%s, orig=%s, moved=%s" % (verb, str(len(orig)), str(len(moved)))
  return orig, moved

def filterZeroShot(dev, train, imgdeps):
  """
  return a dict containing the new split between 0-shotdev and wasted samples.
  """
  newsplit = {"zs": set(), "waste": set()}
  for img in dev:
    label = "waste"
    if imgdeps[img].isZS(train):
      label = "zs"
    newsplit[label].add(img)
  return newsplit
  #return newsplit["zs"], newsplit["waste"]

def filterZeroShotGood(dev, train, imgdeps):
  """
  return a dict containing the new split between 0-shotdev and wasted samples.
  """
  newsplit = {"zs": set(), "waste": set()}
  for img in dev:
    label = "waste"
    if imgdeps[img].isZS(train):
      label = "zs"
    newsplit[label].add(img)
  return newsplit["zs"], newsplit["waste"]

def splitTrainAndDev():
  print "Loading data"
  datasets = ["train.json", "dev.json"]
  full_data = get_joint_set(datasets)
  print "Splitting data"
  datasplit, _ = get_split_2(full_data, desiredZS=12000, ranDevSize=12000)
  print "done splitting data"

  for name, imgset in datasplit.iteritems():
    data = {img: full_data[img] for img in imgset}
    print "Writing file %s" % name
    with open(name + ".json", "w") as f:
      json.dump(data, f)

def splitTrainDevTest():
  """
  Problem: this function will introduce bias in the way that it splits data. For example,
  the "easily chosen" 0-shot images will all be in Dev
  """
  finalSplit = {} # A dict from "datasetname" -> set(images in the dataset)
  print "Loading data"
  datasets = ["train.json", "dev.json", "test.json"]
  full_data = get_joint_set(datasets)
  print "Splitting dev from data"
  devsplit, imgdeps = get_split_2(full_data, desiredZS=12000, ranDevSize=12000)
  print "done splitting dev"

  # Get the dataset minus the training stuff.
  remainingTraining = {k: v for k,v in full_data.iteritems() if k in devsplit["zsTrain"]}

  print "Splitting test from data"
  testsplit, _ = get_split_2(remainingTraining, desiredZS=12000, ranDevSize=12000)
  print "done splitting test"

  print "Pruning waste from zsdev, zstest"
  finalSplit["zsTrain"] = testsplit["zsTrain"]
  finalSplit["zsRanTest"] = testsplit["zsRanDev"] # [sic], naming is suboptimal
  finalSplit["zsRanDev"] = devsplit["zsRanDev"]

  splitDict = filterZeroShot(devsplit["zsDev"], finalSplit["zsTrain"], imgdeps)
  finalSplit["zsDev"] = splitDict["zs"]
  finalSplit["zsDevWaste"] = splitDict["waste"]

  splitDict = filterZeroShot(testsplit["zsDev"], finalSplit["zsTrain"], imgdeps)
  finalSplit["zsTest"] = splitDict["zs"]
  finalSplit["zsTestWaste"] = splitDict["waste"]

  saveDatasets(finalSplit, full_data)

def saveDatasets(splits, full_data):
  """
  Saves splits of the full_data set, as enumerated by the map "splits"
  splits is a dictionary from "dataset_name" -> set(images_in_dataset)
  """
  for name, imgset in splits.iteritems():
    data = {img: full_data[img] for img in imgset}
    print "Writing file %s" % name
    with open(name + ".json", "w") as f:
      json.dump(data, f)

def splitTrainDevTestEvenly():
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
  print "Loading data"
  full_data = get_joint_set(datasets)
  print "Splitting zero-shot from data"
  devsplit, imgdeps = get_split_2(full_data, desiredZS=(devzs+testzs), ranDevSize=(randevsize+rantestsize))
  print "Done splitting zero-shot"

  def addFilteredSubsetsToDict(dictionary, source, condition, passName, failName):
    dictionary[passName] = set()
    dictionary[failName] = set()
    for k in source:
      loc = failName
      if condition(k):
        loc = passName
      dictionary[loc].add(k)

  finalSplit["zsTrain"] = devsplit["zsTrain"]
  ranDev = set(random.sample(devsplit["zsRanDev"], randevsize))
  addFilteredSubsetsToDict(finalSplit, devsplit["zsRanDev"], lambda x: x in ranDev, "zsRanDev", "zsRanTest")

  # TODO can I use the 'filter()' function?
  print "Filtering out zero-shot images."
  split = filterZeroShot(devsplit["zsDev"], finalSplit["zsTrain"], imgdeps)

  ndevzs = len(split["zs"]) / 2
  zsDev = set(random.sample(split["zs"], ndevzs))
  addFilteredSubsetsToDict(finalSplit, split["zs"], lambda x: x in zsDev, "zsDev", "zsTest")
  finalSplit["waste"] = split["waste"]

  saveDatasets(finalSplit, full_data)

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
  full_data = get_joint_set(datasets)
  imgdeps = getImageDeps(getvrn2Imgs(full_data))

  print "getting uniform split"
  init = set(full_data.keys())
  init, finalSplit["zsRanDev"] = get_uniform_split(init, randevsize)
  init, finalSplit["zsRanTest"] = get_uniform_split(init, rantestsize)

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
  zsDevTest, finalSplit["waste"] = filterZeroShotGood(zsDevTestWaste, finalSplit["zsTrain"], imgdeps)
  finalSplit["zsDev"], finalSplit["zsTest"] = get_uniform_split(zsDevTest, len(zsDevTest) / 2)

  for k,v in finalSplit.iteritems():
    print "split %s: %s" % (str(k), str(len(v)))

  saveDatasets(finalSplit, full_data)

def getVerbCounts(filename):
  ret = collections.defaultdict(int)
  full_data = get_joint_set(filename)
  for v in full_data.itervalues():
    ret[v["verb"]] += 1
  return ret

def getDistributionOfVerbs(datasets):
  # get a defaultdict of verb counts for each dataset.

  counts = {}
  for filename in datasets:
    counts[filename] = getVerbCounts(filename)

  # Combine the counts into a final dict.
  finalcounts = collections.defaultdict(lambda : collections.defaultdict(int))
  for filename, wordcount in counts.iteritems():
    for word, frequency in wordcount.iteritems():
      finalcounts[word][filename] += frequency
  return finalcounts

def print_verb_distribution(distribution):
  order = ["zsTrain", "zsDev", "zsTest", "zsRanDev", "zsRanTest", "waste"]
  s = ""
  for o in order:
    s += "%s   " % o
  print s
  for word, stats in distribution.iteritems():
    s = ""
    for o in order:
      s += "%d  " % stats["%s.json" % o]
    print s

def getDistStub():
  datasets = ["zsTrain.json", "zsRanDev.json", "zsRanTest.json", "zsDev.json", "zsTest.json", "waste.json"]
  return getDistributionOfVerbs(datasets)

def getDifferers(datasets):
  """
  Determines how many verb-roles differ between each pair of images that
  share a verb class.
  """
  #data = get_joint_set(["train.json", "dev.json"])
  data = get_joint_set(datasets)
  #data = get_joint_set(["testfunc.json"])
  vrn2Imgs = getvrn2Imgs(data)
  imgdeps = getImageDeps(vrn2Imgs)

  differCounts = collections.defaultdict(int)

  verbsets = collections.defaultdict(list)
  for name,imgdep in imgdeps.iteritems():
    verbsets[imgdep.verb].append(name)
  classnum = 0
  totclasses = len(verbsets)
  for verb, names in verbsets.iteritems():
    classnum += 1
    print "===> Considering Verb Class %d of %d" % (classnum, totclasses)
    for i in range(len(names)):
      for j in range(i):
        img1 = imgdeps[names[i]]
        img2 = imgdeps[names[j]]
        ndiff = img1.numDifferentLabelings(img2)
        differCounts[ndiff] += 1
  print str(differCounts)
  return differCounts

def getnn2vr2ndiffs(datasets):
  """
  Determines how many verb-roles differ between each pair of images that
  share a verb class.
  """
  #data = get_joint_set(["train.json", "dev.json"])
  data = get_joint_set(datasets)
  #data = get_joint_set(["testfunc.json"])
  vrn2Imgs = getvrn2Imgs(data)
  imgdeps = getImageDeps(vrn2Imgs)

  nn2vr2ndiffs = {}

  verbsets = collections.defaultdict(list)
  for name,imgdep in imgdeps.iteritems():
    verbsets[imgdep.verb].append(name)
  classnum = 0
  totclasses = len(verbsets)
  for verb, names in verbsets.iteritems():
    classnum += 1
    print "===> Considering Verb Class %d of %d" % (classnum, totclasses)
    for i in range(len(names)):
      for j in range(len(names)):
        vr = (verb, role)
        key = (names[i], names[j])
        nn2vr2diffs[(verb, role)]
        nn2ndiffs[key] = img1.numDifferentLabelings(img2)
  print str(differCounts)
  return differCounts

def get_names(filename, out):
  data = get_joint_set(filename)
  output = "\n".join(data.keys())
  with open(out, "w") as f:
    f.write(output)

def get_all_names():
  names = ["zsTrain.json", "zsDev.json", "zsTest.json", "zsRanDev.json", "zsRanTest.json", "waste.json"]
  for name in names:
    outname = name.replace(".json", ".txt")
    get_names(name, outname)

def main():
  trainDeps, devDeps, imgdeps = get_split(["train.json", "dev.json"], 0.1)

  # Count the number of 0-shot images we produced.
  nZero = 0
  nZero = countZS(devDeps, trainDeps, imgdeps)
  #for img in devDeps:
  #  if imgdeps[img].isZS(trainDeps):
  #    nZero += 1

  # Now, you should have moved enough images.
  print "Train size: %d" % len(trainDeps)
  print "Dev size: %d" % len(devDeps)
  print "nZeroShot: %d" % nZero
  return 0
