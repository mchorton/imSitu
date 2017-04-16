import data_utils as du
import collections
import math
import json
import itertools
import os
import cPickle
import splitutils as ut
import itertools as it
import copy
import numpy as np
import tqdm
import utils.mylogger as logging

def threegramProb(key, givenIndex, wildcardCounts):
  #exit(1) # TODO deprecated.
  num = wildcardCounts[key]
  wildkey = ['*'] * len(key)
  wildkey[givenIndex] = key[givenIndex]
  den = wildcardCounts[tuple(wildkey)]
  return 1. * num / den

def getWildcardKey(key, givenIndices):
  return tuple([key[val] if val in givenIndices else '*' for val in range(len(key))])

def allgramProb(key, givenIndex, wildcardCounts, weights):
  exit(1) # TODO deprecated. Not totally correct.
  """
  This truly is some sort of allgram prob. measure. # TODO but it's wrong!
  Weights: a dict from number_of_fixed_classes -> weight for those classes.
  Note that the weight will be divided evenly among every combination of
  choices that fix that many elements.
  Actually, it's a dict from "nRoles" to the above thing.
  Also, at least 1 class is always fixed (the conditional), so no key should
  be below 1.
  """
  nRoles = -1
  for k in wildcardCounts:
    nRoles = len(k)
    break

  num = 0
  den = wildcardCounts[getWildcardKey(key, [givenIndex])]

  myweights = weights[nRoles]

  for ps in ut.powerset(range(nRoles)):
    if givenIndex not in ps:
      continue
    #mykey = tuple([key[val] if val not in replaceIndices else '*' for val in range(len(key))])
    mykey = getWildcardKey(key, ps)
    num += 1. * wildcardCounts[mykey] * myweights[len(ps)] / ut.ncr(nRoles, len(ps))
  return 1. * num / den

def allgramHellingerDist(vrProb, role, noun1, noun2):
  total = 0
  # Gets the distance between two nouns in a given role.
  roleIndex = vrProb.order.index(role)
  keyset = [k for k,v in vrProb.counts.iteritems() if k[roleIndex] in (noun1, noun2)]
  keyset = [list(k) for k in keyset]
  for k in keyset:
    k[roleIndex] = 0
  # Remove unique elements. Lists are unhashable, tuples are immutable. WHY
  newkeyset = []
  for key in keyset:
    if key not in newkeyset:
      newkeyset.append(key)
  keyset = newkeyset

  # get normalizations.
  normalization1 = 0
  normalization2 = 0
  for k,v in vrProb.counts.iteritems():
    if k[roleIndex] == noun1:
      normalization1 += v
    if k[roleIndex] == noun2:
      normalization2 += v

  # Finally, calculate values.
  for key in keyset:
    key[roleIndex] = noun1
    prob1 = vrProb.getProb(tuple(key), roleIndex)
    key[roleIndex] = noun2
    prob2 = vrProb.getProb(tuple(key), roleIndex)
    if prob1 > 1.:
      print key
      exit(1)
    if prob2 > 1.:
      print key
      exit(1)
    total += (math.sqrt(prob1) - math.sqrt(prob2)) ** 2
  return math.sqrt(total) / math.sqrt(2)

class VRProb():
  def __init__(self, verb, pfunc=threegramProb, dfunc = allgramHellingerDist):
    self.counts = None
    self.order = None # This is the order that the counts map's keys are in.
    self.pfunc=pfunc
    self.distFunc = dfunc
    self.verb = verb
  def getVerb(self):
    return self.verb
  def normalize(self, frame):
    ret = []
    for o in self.order:
      ret.append(frame[o])
    return tuple(ret)
  def add(self, frames):
    if self.counts is None or self.order is None:
      self.order = tuple(sorted(frames[0].keys()))
      self.counts = collections.defaultdict(int)
    for frame in frames:
      self.counts[self.normalize(frame)] += 1
    self.makeWildcardCounts() # woops, shouldn't do this so often...
  @property
  def nRoles(self):
    return len(self.order)
  def makeWildcardCounts(self):
    # Makes counts that have wildcard keys as well.
    self.wildcardCounts = collections.defaultdict(int)
    for k,v in self.counts.iteritems():
      for replaceIndices in ut.powerset(range(self.nRoles)):
        # Create a tuple, but with all the "replaceIndices" changed.
        mykey = tuple([k[val] if val not in replaceIndices else '*' for val in range(len(k))]) # TODO carefully replace w/ other func.
        self.wildcardCounts[mykey] += v

  def getProb(self, key, givenIndex):
    prob = self.pfunc(key, givenIndex, self.wildcardCounts)
    if prob > 1:
      logging.getLogger(__name__).info("bad prob=%s for key: %s" % (str(prob), str(key)))
    return prob

  def match(self, test, target, roleIndex): # TODO unused.
    # Determine whether 'test' matches 'target' everywhere except at roleindex
    return all([test[i] == target[i] for i in range(len(test)) if i != roleIndex])
  def getDistance(self, role, noun1, noun2):
    return self.distFunc(self, role, noun1, noun2)

class VRProbDispatcher():
  def __init__(self, data, chosen = None, pfunc=threegramProb, dfunc = allgramHellingerDist):
    self.data = data
    self.pfunc=pfunc
    self.dfunc = dfunc
    self.verbVRPMap = {}
    if not chosen:
      return
    self.addComparisons(chosen)
  def addComparisons(self, chosen):
    # Clear out the chosen verbs, we're recomputing them
    for c in chosen:
      self.verbVRPMap.pop(c, None)
    # Now, populate with new values.
    for k,v in self.data.iteritems():
      verb = v["verb"]
      if verb not in chosen:
        continue
      if verb not in self.verbVRPMap:
        self.verbVRPMap[verb] = VRProb(verb, pfunc=self.pfunc, dfunc=self.dfunc)
      self.verbVRPMap[verb].add(v["frames"])
  def getDistance(self, verb, *args):
    return self.verbVRPMap[verb].getDistance(*args)
  def getAllSimilarities(self, verb, role):
    if verb not in self.verbVRPMap:
      self.addComparisons([verb])
    nouns = getAllNounsForVR(verb, role, self.data)
    return getSimilarities(verb, role, nouns, self)

def getAllNounsForVR(verb, role, data):
  nouns = set()
  for k,v in data.iteritems(): 
    if v["verb"] != verb:
      continue
    for frame in v["frames"]:
      nouns.add(frame[role])
  return nouns

def getSimilarities(verb, role, nouns, vrProbs):
  similarities = {}
  for noun1 in nouns:
    for noun2 in nouns:
      if noun2 < noun1:
        continue # arbitrary
      similarities[(noun1, noun2)] = vrProbs.getDistance(verb, role, noun1, noun2)
  return similarities

# TODO delete this?
def runGetSimilarities(datasetLoc):
  data = du.get_joint_set(datasetLoc)

  chosen = ["riding"]
  vrProbs = VRProbDispatcher(data, chosen)

  # Get a few nouns for the given role
  nouns = getAllNounsForVR("riding", "vehicle", data)

  similarities = getSimilarities("riding", "vehicle", nouns, vrProbs)
  desired = [(du.decodeNoun(k[0]), du.decodeNoun(k[1]), v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  return desired

def getNounImgs(verb, role, noun, vrn2Images):
  return vrn2Images.get((verb, role, noun), set())

def vecDist(vrProb, role, noun1, noun2):
  roleIndex = vrProb.order.index(role)
  # get all the nouns associated with each role.
  allRoleNouns = map(set, list(zip(*vrProb.counts.keys())))
  noun1Prob = []
  noun2Prob = []
  for n, roleNouns in enumerate(allRoleNouns):
    if n == roleIndex:
      continue
    for noun in roleNouns:
      key1 = ['*'] * vrProb.nRoles
      key1[roleIndex] = noun1
      den1 = vrProb.wildcardCounts[tuple(key1)]
      key1[n] = noun
      num1 = vrProb.wildcardCounts[tuple(key1)]
      noun1Prob.append(1. * num1 / den1)

      key2 = ['*'] * vrProb.nRoles
      key2[roleIndex] = noun2
      den2 = vrProb.wildcardCounts[tuple(key2)]
      key2[n] = noun
      num2 = vrProb.wildcardCounts[tuple(key2)]
      noun2Prob.append(1. * num2 / den2)
  return np.linalg.norm(np.array(noun1Prob) - np.array(noun2Prob))

class WholisticSimCalc(object):
  def __init__(self, vrn2Imgs):
    self.vrn2Imgs = vrn2Imgs
    self.n2vr2Imgs = du.getn2vr2Imgs(self.vrn2Imgs)
    # Precache some oft-needed objects.
    self.n2ImgSet = { n: set((img for imgs in vr2Imgs.itervalues() for img in imgs)) for n, vr2Imgs in self.n2vr2Imgs.iteritems()}
    self.n2vrSet = { n: set(vr2Imgs.keys()) for n, vr2Imgs in self.n2vr2Imgs.iteritems()}

    # Make a structure to cache similarity computations.
    self.simDbg = {}

  def wholisticSimilarity(self, vrProb, role, noun1, noun2):
    """
    For intermediate metrics, a score of 1 indicates a good pair, a score of
    0 indicates a poor pair. So, no reason to negate it!
    """
    vsim = math.exp(-vecDist(vrProb, role, noun1, noun2))

    # Get similarity across roles.
    n1roleSet = self.n2vrSet[noun1]
    n2roleSet = self.n2vrSet[noun2]
    arsim = 1. * len(n1roleSet.intersection(n2roleSet)) / len(n1roleSet.union(n2roleSet))

    # Get similarity across the image set.
    n1Imgs = self.n2ImgSet[noun1]
    n2Imgs = self.n2ImgSet[noun2]
    aidif = 1. - (1. * len(n1Imgs.intersection(n2Imgs)) / len(n1Imgs.union(n2Imgs)))

    ret = 1. * vsim * arsim * aidif
    key = (noun1, noun2)
    if key not in self.simDbg:
      self.simDbg[key] = []
    self.simDbg[(noun1, noun2)].append({"vsim": vsim, "arsim": arsim, "aidif": aidif, "ret": ret, "verb": vrProb.getVerb(), "role": role})
    return ret

  def getWSLam(self):
    return lambda vrProb, role, noun1, noun2: self.wholisticSimilarity(vrProb, role, noun1, noun2)

def getnn2vr2score(v2r2nn2score):
  ret = {}
  for v,rest1 in v2r2nn2score.iteritems():
    for r,rest2 in rest1.iteritems():
      for nn, score in rest2.iteritems():
        if nn not in ret:
          ret[nn] = {}
        ret[nn][(v,r)] = score
  return ret

class SimilaritiesListCalculator(object):
  def __init__(self, dirName, **kwargs):
    self.dirName = dirName
    self.kwargs = kwargs
    self.simList = None
  @property
  def outname(self):
    return self.dirName + "chosen_pairs_%s.json" % str(sorted(self.kwargs.iteritems()))
  def getSimilaritiesList(self):
    if self.simList is None:
      if os.path.isfile(self.outname):
        self.simList = json.load(open(self.outname))
      else: # We have to calculate it, womp womp
        self.simList = getSimilaritiesList(self.dirName, **self.kwargs)
        json.dump(self.simList, open(self.outname, "w"))
    return self.simList

def getAveragedRankings(distdir, data):
  """
  distdir - directory with saved distance data
  data - the actual dataset (usually from zsTrain.json)
  """
  logging.getLogger(__name__).info("Getting averaged rankings")
  v2r = du.getv2r(data)
  v2r2nn2score = {}
  logging.getLogger(__name__).info("loading data")
  v2r2nn2score = {verb: cPickle.load(open("%s%s.pik" % (distdir, verb), "r")) for verb in v2r}

  logging.getLogger(__name__).info("getting nn2vr2score")
  nn2vr2score = getnn2vr2score(v2r2nn2score)
  cPickle.dump(nn2vr2score, open("%s%s.pik" % (distdir, "nn2vr2score"), "w"))

  logging.getLogger(__name__).info("combining values")
  flatAvg = {nn: np.mean(vr2score.values()) for nn, vr2score in nn2vr2score.iteritems()}
  return flatAvg

# TODO this should take some logging object, and output HTML that shows stats
# about what happened during the run.
def getSimilaritiesList(dirName, datasetLoc, thresh=2., freqthresh = 10, blacklistprs = [set(["man", "woman"])], bestNounOnly = True, noThreeLabel = True, noOnlyOneRole = True, strictImgSep = True):
  """
  Get output like:
  [[noun1, noun2, similarity, img1, img2, tuple(verb, role)], ...]
  Make HTML that shows one image pair per line, ordered by distance between the
  images in similarity space.
  freqthresh: if a noun occurs freqthresh or fewer times, it'll be excluded.
  blacklistprs: each pair (n1, n2) that matches 
  """
  train = du.get_joint_set(datasetLoc)

  similarities = getAveragedRankings(dirName, train)
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired

  logging.getLogger(__name__).info("Getting vrn2Imgs")
  vrn2Imgs = du.getvrn2Imgs(train)
  logging.getLogger(__name__).info("vrn2Imgs is: %s" % str(vrn2Imgs))
  logging.getLogger(__name__).info("Getting n2vr2Imgs")
  n2vr2Imgs = du.getn2vr2Imgs(vrn2Imgs)

  toShow2 = [] # list like (n1,n2,sim,img1,img2)

  logging.getLogger(__name__).info("Total num sims: %d" % len(similarities))

  imgdeps = du.getImageDeps(vrn2Imgs)

  # Get the number of images in which each noun occurs.
  freqs = du.getFrequencies(imgdeps)

  im2vr2bestnoun = du.get_im2vr2bestnoun(train)
  examiner = du.DataExaminer()
  examiner.analyze(train)

  stats = collections.defaultdict(int)
  nPassSame = 0
  nPassDiff = 0
  nNotBestLabel = 0 # This number is meaningless since I didn't check roles before incrementing. Oh well.
  logging.getLogger(__name__).info("looping...")
  for i, (n1, n2, sim) in tqdm.tqdm(enumerate(similarities), total=len(similarities), desc="Calculating similarities object..."):
    #logging.getLogger(__name__).info("i=%d" % i)
    if n1 == "" or n2 == "":
      continue
    if set([du.decodeNoun(n1),du.decodeNoun(n2)]) in blacklistprs:
      continue
    if freqs[n1] < freqthresh or freqs[n2] < freqthresh:
      continue
    if noOnlyOneRole and (len(n2vr2Imgs[n1]) <= 1 or len(n2vr2Imgs[n2]) <= 1):
      continue
    for vr, imgs in n2vr2Imgs[n1].iteritems():
      firstset = imgs
      secondset = n2vr2Imgs[n2].get(vr, [])
      for one,two in it.product(firstset, secondset):
        #bestOneLabel = examiner.getBestNoun(one, vr)
        #bestTwoLabel = examiner.getBestNoun(two, vr)
        bestOneLabel = im2vr2bestnoun[one][vr]
        bestTwoLabel = im2vr2bestnoun[two][vr]
        if bestNounOnly and (n1 != bestOneLabel or n2 != bestTwoLabel):
          nNotBestLabel += 1
          continue
        if noThreeLabel and (len(set(examiner.getNouns(one, vr))) == 3 or len(set(examiner.getNouns(two, vr))) == 3):
          stats["n3lab"] += 1
          continue
        dl = imgdeps[one].differentLabelings(imgdeps[two], strictImgSep, examiner)
        if len(dl) == 1:
          if dl[0] == vr:
            nPassSame += 1
            toShow2.append([n1, n2, sim, one, two, tuple(vr)])
          else:
            nPassDiff += 1
  logging.getLogger(__name__).info("nPassSame=%s" % str(nPassSame))
  logging.getLogger(__name__).info("nPassDiff=%s" % str(nPassDiff))
  logging.getLogger(__name__).info("nNotBestLabel=%s" % str(nNotBestLabel))
  logging.getLogger(__name__).info("stats=%s" % str(stats))

  logging.getLogger(__name__).info("There are %d valid pairs" % len(toShow2))
  toShow2 = [t for t in toShow2 if t[2] < thresh]
  logging.getLogger(__name__).info(
      "Cutoff thresh %f: there are %d valid pairs" % (thresh, len(toShow2)))
  logging.getLogger(__name__).info("Expanding (n1, n2) => (n1, n2), (n2, n1)")
  nextShow = []
  # Generate reverse pairs.
  for elem in tqdm.tqdm(
      toShow2, total=len(toShow2), desc="Generating reverse pairs"):
    nextShow.append(elem)
    newelem = copy.deepcopy(elem)
    newelem[0] = elem[1]
    newelem[1] = elem[0]
    newelem[3] = elem[4]
    newelem[4] = elem[3]
    nextShow.append(newelem)
  toShow2 = nextShow

  return toShow2 # Probably should return an object...
