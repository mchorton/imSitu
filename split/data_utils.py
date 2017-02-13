import json
import copy
import collections
import random

class ImgDep():
  def __init__(self, name):
    self.name = name
    # Map from verb-role to set of image names.
    self.deps = {} # TODO add_deps function?
    self.role2Nouns = {}
    self.role2NounList = {}
  def __str__(self):
    return "(%s, %s)" % (str(self.name), str(self.deps))
  def __repr__(self):
    return self.__str__()

  def isZS(self, trainImgDeps):
    """ 
    Check whether this image is 0-shot 
    """
    return len(self.minNeededForZS(trainImgDeps)) == 0

  @property
  def verb(self):
    """ Get the name of the verb represented in this image. """
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

  def differentLabelings(self, otherImgDep):
    """
    Find the roles over which these differ.
    Return as a list: [(verb, role1), (verb, role2), ...]
    """
    diffRoles = []
    if self.verb is None or self.verb != otherImgDep.verb:
      return None
    numDifferences = 0
    for k,v in self.role2Nouns.iteritems():
      if len(otherImgDep.role2Nouns[k] & v) == 0:
        diffRoles.append(k)
    return diffRoles

  def numDifferentLabelings(self, otherImgDep):
    ret = self.differentLabelings(otherImgDep)
    if ret is None:
      return None
    return len(ret)

def getFrequencies(imgdeps):
  ret = collections.defaultdict(int)
  for img,dep in imgdeps.iteritems():
    seen = set([n for sublist in dep.role2Nouns.values() for n in sublist])
    for elem in seen:
      ret[elem] += 1
  return ret

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
      imgdeps[img].role2NounList[vr] = imgdeps[img].role2NounList.get(vr, []) + [vrn[2]]
  return imgdeps

def get_joint_set(datasets):
  if  isinstance(datasets, str):
    datasets = [datasets]
  train = {}
  for dataset in datasets:
    train.update(json.load(open(dataset)))
  return train

def countZS(desiredSet, trainDeps, imgdeps):
  """
  Find how many images in desiredSet are zero shot with respect to trainDeps
  (e.g. desiredSet is, e.g., the test set)
  """
  count = 0
  nZero = 0
  for img in desiredSet:
    count += 1
    if imgdeps[img].isZS(trainDeps):
      nZero += 1
  return nZero

def get_uniform_split(init, desiredOtherCount):
  """
  Split the data in 'init' into two sets, with sizes 
  (len(init)-desiredOtherCount) and desiredOtherCount, respectively
  """
  ranSelection = set(random.sample(init, desiredOtherCount))
  newInit = set([k for k in init if k not in ranSelection])
  return newInit, ranSelection

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
  return newsplit["zs"], newsplit["waste"]

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

_nouncache = None
def decodeNoun(noun):
  global _nouncache
  if noun == "":
    return ""
  if _nouncache is None:
    _nouncache = json.load(open("imsitu_space.json"))["nouns"]
  return _nouncache.get(noun, {"gloss": [noun]})["gloss"][0]

def decodeNouns(*args):
  return tuple(map(decodeNoun, args))
