# 1: get dependency structure.
# 2: get 

# Note: when you move one image, you need to check how many images (in B)
# became 0-shot. You also need to consider their desired pairings.

import json
import copy
import collections

MAXTHRESH=10

class ImgDep():
  def __init__(self, name):
    self.name = name
    # Map from verb-role to set of image names.
    self.deps = {} # TODO add_deps function?
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
  for vrn, imgs in vrn2Imgs.iteritems():
    for img in imgs:
      if img not in imgdeps:
        imgdeps[img] = ImgDep(img)
      vr = vrn[:2]
      imgdeps[img].deps[vr] = imgdeps[img].deps.get(vr, set()) | imgs
  return imgdeps

def get_split(datasets, devsize):
  """ 
  Gets the desired splits of the data.
  @param datasets a list of datasets from which data will be merged, or a single filename
  @param devsize the % of data to put in the dev set
  """
  if  isinstance(datasets, str):
    datasets = [datasets]
  train = {}
  for dataset in datasets:
    train.update(json.load(open(dataset)))
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

def main():
  trainDeps, devDeps, imgdeps = get_split(["train.json", "dev.json"], 0.3)

  # Count the number of 0-shot images we produced.
  nZero = 0
  for img in devDeps:
    if imgdeps[img].isZS(trainDeps):
      nZero += 1

  # Now, you should have moved enough images.
  print "Train size: %d" % len(trainDeps)
  print "Dev size: %d" % len(devDeps)
  print "nZeroShot: %d" % nZero
  return 0
