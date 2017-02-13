import data_utils as du
import collections
import math
import json
import itertools
import os
import numpy
import cPickle
import utils as ut

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
  def __init__(self, pfunc=threegramProb, dfunc = allgramHellingerDist):
    self.counts = None
    self.order = None # This is the order that the counts map's keys are in.
    self.pfunc=pfunc
    self.distFunc = dfunc

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
      print "bad prob=%s for key: %s" % (str(prob), str(key))
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
        self.verbVRPMap[verb] = VRProb(pfunc=self.pfunc, dfunc=self.dfunc)
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

def runGetSimilarities():
  datasets = ["zsTrain.json"]
  data = du.get_joint_set(datasets)

  chosen = ["riding"]
  vrProbs = VRProbDispatcher(data, chosen)

  # Get a few nouns for the given role
  nouns = getAllNounsForVR("riding", "vehicle", data)

  similarities = getSimilarities("riding", "vehicle", nouns, vrProbs)
  desired = [(decodeNoun(k[0]), decodeNoun(k[1]), v) for k,v in similarities.iteritems()]
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
  return numpy.linalg.norm(numpy.array(noun1Prob) - numpy.array(noun2Prob))

def getnn2vr2score(v2r2nn2score):
  ret = {}
  for v,rest1 in v2r2nn2score.iteritems():
    for r,rest2 in rest1.iteritems():
      for nn, score in rest2.iteritems():
        if nn not in ret:
          ret[nn] = {}
        ret[nn][(v,r)] = score
  return ret
