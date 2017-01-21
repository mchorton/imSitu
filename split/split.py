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
  ret["zsranDev"] = ranDev

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
        #if len(devDeps) > nDev:
        #  cont = False
        #  break
    # count the number of 0-shot images
    nZS = countZS(devDeps, trainDeps, imgdeps)
    print "after iteration: obtained %s 0-shot, desire %s" % (str(nZS), desiredZS)
    if nZS >= desiredZS or len(trainDeps) == 0:
      cont = False
      break
    if not anyMoved:
      movethresh += 1

  ret["zstrain"] = trainDeps
  ret["zsDev"] = devDeps

  return ret, imgdeps

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
