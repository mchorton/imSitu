import split as split
import collections
import math
import json
import itertools
import os
import numpy
import cPickle

import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def threegramProb(key, givenIndex, wildcardCounts):
  num = wildcardCounts[key]
  wildkey = ['*'] * len(key)
  wildkey[givenIndex] = key[givenIndex]
  den = wildcardCounts[tuple(wildkey)]
  return 1. * num / den

def getWildcardKey(key, givenIndices):
  return tuple([key[val] if val in givenIndices else '*' for val in range(len(key))])

def allgramProb(key, givenIndex, wildcardCounts, weights):
  """
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

  for ps in powerset(range(nRoles)):
    if givenIndex not in ps:
      continue
    #mykey = tuple([key[val] if val not in replaceIndices else '*' for val in range(len(key))])
    mykey = getWildcardKey(key, ps)
    num += 1. * wildcardCounts[mykey] * myweights[len(ps)] / ncr(nRoles, len(ps))
  return 1. * num / den


  num = wildcardCounts[key]
  wildkey = ['*'] * len(key)
  wildkey[givenIndex] = key[givenIndex]
  den = wildcardCounts[tuple(wildkey)]
  return 1. * num / den

def allgramHellingerDist(self, role, noun1, noun2):
  total = 0
  # Gets the distance between two nouns in a given role.
  roleIndex = self.order.index(role)
  keyset = [k for k,v in self.counts.iteritems() if k[roleIndex] in (noun1, noun2)]
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
  for k,v in self.counts.iteritems():
    if k[roleIndex] == noun1:
      normalization1 += v
    if k[roleIndex] == noun2:
      normalization2 += v

  # Finally, calculate values.
  for key in keyset:
    key[roleIndex] = noun1
    prob1 = self.getProb(tuple(key), roleIndex)
    key[roleIndex] = noun2
    prob2 = self.getProb(tuple(key), roleIndex)
    if prob1 > 1.:
      print key
      exit(1)
    if prob2 > 1.:
      print key
      exit(1)
    """
    key[roleIndex] = noun1
    prob1 = 1. * self.counts[tuple(key)] / normalization1
    key[roleIndex] = noun2
    prob2 = 1. * self.counts[tuple(key)] / normalization2
    """
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
      for replaceIndices in powerset(range(self.nRoles)):
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

_nouncache = None
def decodeNoun(noun):
  global _nouncache
  if noun == "":
    return ""
  if _nouncache is None:
    _nouncache = json.load(open("imsitu_space.json"))["nouns"]
  return _nouncache.get(noun, {"gloss": [noun]})["gloss"][0]

def decodeNouns(*args):
  tuple(map(decodeNoun, args))

def runGetSimilarities():
  datasets = ["zsTrain.json"]
  data = split.get_joint_set(datasets)

  chosen = ["riding"]
  vrProbs = VRProbDispatcher(data, chosen)

  # Get a few nouns for the given role
  nouns = getAllNounsForVR("riding", "vehicle", data)

  similarities = getSimilarities("riding", "vehicle", nouns, vrProbs)
  desired = [(decodeNoun(k[0]), decodeNoun(k[1]), v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  return desired

def mytostr(*args):
  return tuple(["" if arg is None else arg for arg in args])

class HtmlTable():
  def __init__(self):
    self.rows = []
    self.header = """ <!DOCTYPE html>
    <html>
    <head>
    <style>
    table, th, td {
          border: 1px solid black;
    }
    td { white-space:pre }
    </style>
    </head>
    <body>
    """
    self.footer = """</body>
    </html>"""
    self.imPerRow = 4
  def addRow(self, urls1, urls2, label):
    self.rows.append([urls1, urls2, label])
  def __str__(self):
    output = ""
    output += self.header
    output += "<table>"
    for row in self.rows:
      output += "<tr>\n"
      for n, elem in enumerate(row):
        output += "<th>"
        if n in (0,1):
          output += "\n".join(map("".join, map(mytostr, *[iter(elem)]*self.imPerRow)))
        else:
          output += "%s" % elem
        output += "</th>\n" # TODO end th?
      output += "</tr>\n"
    output += "</table>"
    output += self.footer
    return output

# TODO make this local
def getImgUrl(name):
  return '<img src="https://s3.amazonaws.com/my89-frame-annotation/public/images_256/%s">' % (name)

def getImgUrls(names):
  ret = set()
  for k in names:
    ret.add(getImgUrl(k))
  #return "".join(list(ret))
  return ret

def getNounImgs(verb, role, noun, vrn2Images):
  return vrn2Images.get((verb, role, noun), set())

def generateHTML(verb, role, vrProbs, vrn2Imgs, outdir):
  chosen = [verb]
  similarities = vrProbs.getAllSimilarities(verb, role)
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired

  getHTMLHelper(similarities, verb, role, vrn2Images, outdir)

def generateHTMLHelper(similarities, verb, role, vrn2Imgs, outdir):
  """
  similarities - a list like [(noun1, noun2, similarity_score as a float), (...), ...]
  """
  htmlTable = HtmlTable()

  for k in similarities:
    img1Urls = getImgUrls(getNounImgs(verb, role, k[0], vrn2Imgs))
    img2Urls = getImgUrls(getNounImgs(verb, role, k[1], vrn2Imgs))
    if img1Urls == img2Urls or 0 in map(len, (img1Urls, img2Urls)):
      continue
    htmlTable.addRow(img1Urls, img2Urls, "d(%s,%s)=%.4f" % (decodeNoun(k[0]), decodeNoun(k[1]), k[2]))
  with open(outdir + "%s-%s.html" % (verb, role), "w") as f:
    f.write(str(htmlTable))

# TODO: jumping-agent (which has an 'empty-set' sign) has some bug...
def generateAllHTML(loc):
  chosen = [("riding", "vehicle"), ("crawling", "agent"), ("distracting", "place"), ("jumping", "obstacle"), ("jumping", "agent"), ("jumping", "destination")]

  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data)

  weights = {}
  weights[1] = collections.defaultdict(float)
  weights[1][1] = 1
  weights[2] = collections.defaultdict(float)
  weights[2][1] = 0.5
  weights[2][2] = 0.5
  weights[3] = collections.defaultdict(float)
  weights[3][1] = 0.33
  weights[3][2] = 0.33
  weights[3][3] = 0.34
  weights[4] = collections.defaultdict(float)
  weights[4][1] = 0.25
  weights[4][2] = 0.25
  weights[4][3] = 0.25
  weights[4][4] = 0.25
  weights[5] = collections.defaultdict(float)
  weights[5][1] = 0.2
  weights[5][2] = 0.2
  weights[5][3] = 0.2
  weights[5][4] = 0.2
  weights[5][5] = 0.2
  """
  weights = {}
  weights[1] = collections.defaultdict(float)
  weights[1][1] = 1
  weights[2] = collections.defaultdict(float)
  weights[2][1] = 0.
  weights[2][2] = 1.
  weights[3] = collections.defaultdict(float)
  weights[3][1] = 0.
  weights[3][2] = 0.
  weights[3][3] = 1.
  weights[4] = collections.defaultdict(float)
  weights[4][1] = 0.
  weights[4][2] = 0.
  weights[4][3] = 0.
  weights[4][4] = 1.
  weights[5] = collections.defaultdict(float)
  weights[5][1] = 0.
  weights[5][2] = 0.
  weights[5][3] = 0.
  weights[5][4] = 0.
  weights[5][5] = 1.
  """

  myprob = lambda x,y,z: allgramProb(x,y,z,weights)

  vrProbs = VRProbDispatcher(data, chosen, myprob)
  #def allgramProb(key, givenIndex, wildcardCounts, weights):

  for verb, role in chosen:
    print "Generating html for %s-%s" % (verb, role)
    generateHTML(verb, role, vrProbs, vrn2Imgs, loc)

# TODO: jumping-agent (which has an 'empty-set' sign) has some bug...
def generateAllDistOnly2Fixed(loc):
  chosen = [("riding", "vehicle"), ("crawling", "agent"), ("distracting", "place"), ("jumping", "obstacle"), ("jumping", "agent"), ("jumping", "destination")]
  # TODO get all verb-roles.

  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data)

  weights = {}
  weights[1] = collections.defaultdict(float)
  weights[1][1] = 1 # Whatever...
  weights[2] = collections.defaultdict(float)
  weights[2][2] = 1.
  weights[3] = collections.defaultdict(float)
  weights[3][2] = 1.
  weights[4] = collections.defaultdict(float)
  weights[4][2] = 1.
  weights[5] = collections.defaultdict(float)
  weights[5][2] = 1.
  weights[6] = collections.defaultdict(float)
  weights[6][2] = 1.
  weights[7] = collections.defaultdict(float)
  weights[7][2] = 1.

  myprob = lambda x,y,z: allgramProb(x,y,z,weights)

  vrProbs = VRProbDispatcher(data, chosen, myprob)

  saveAllDistances(data, vrProbs, loc)

def saveAllDistances(data, vrProbs, outdir):
  v2r = split.getv2r(data)
  for i, (verb,roles) in enumerate(v2r.iteritems()):
    print "Considering verb %d/%d" % ((i + 1), len(v2r))
    outfile = "%s%s.pik" % (outdir, verb)
    if os.path.isfile(outfile):
      print "...Found precached results for verb '%s'" % (verb)
      continue
    print "...Computing all distances for verb '%s'..." % verb
    vrn2dist = {role: vrProbs.getAllSimilarities(verb, role) for role in roles}
    print "...Saving distances for verb '%s' to '%s'..." % (verb, outfile)
    cPickle.dump(vrn2dist, open(outfile, "w"))
    print "...done."

def twoFixedStub():
  generateAllDistOnly2Fixed("data/twoFixedDist/")

def vecStyleStub():
  #generateAllDistVecStyle("data/vecStyle/")
  #getAveragedRankings("data/vecStyle/")
  generateHTMLForExp("data/vecStyle/")

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
      #print "p1=%.3f" % noun1Prob[-1]
      #print "p2=%.3f" % noun2Prob[-1]

  return numpy.linalg.norm(numpy.array(noun1Prob) - numpy.array(noun2Prob))



def generateAllDistVecStyle(loc):
  chosen = [("riding", "vehicle"), ("crawling", "agent"), ("distracting", "place"), ("jumping", "obstacle"), ("jumping", "agent"), ("jumping", "destination")]
  # TODO get all verb-roles.

  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data)

  weights = {}
  weights[1] = collections.defaultdict(float)
  weights[1][1] = 1 # Whatever...
  weights[2] = collections.defaultdict(float)
  weights[2][2] = 1.
  weights[3] = collections.defaultdict(float)
  weights[3][2] = 1.
  weights[4] = collections.defaultdict(float)
  weights[4][2] = 1.
  weights[5] = collections.defaultdict(float)
  weights[5][2] = 1.
  weights[6] = collections.defaultdict(float)
  weights[6][2] = 1.
  weights[7] = collections.defaultdict(float)
  weights[7][2] = 1.

  myprob = lambda x,y,z: allgramProb(x,y,z,weights)

  vrProbs = VRProbDispatcher(data, chosen, myprob, vecDist)

  saveAllDistances(data, vrProbs, loc)






def getnn2vr2score(v2r2nn2score):
  ret = {}
  for v,rest1 in v2r2nn2score.iteritems():
    for r,rest2 in rest1.iteritems():
      for nn, score in rest2.iteritems():
        if nn not in ret:
          ret[nn] = {}
        ret[nn][(v,r)] = score
  return ret

def getAveragedRankings(directory):
  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)
  v2r = split.getv2r(data)
  # 
  v2r2nn2score = {}
  print "loading data..."
  v2r2nn2score = {verb: cPickle.load(open("%s%s.pik" % (directory, verb), "r")) for verb in v2r}
  print "...done loading data"

  print "getting nn2vr2score"
  nn2vr2score = getnn2vr2score(v2r2nn2score)
  cPickle.dump(nn2vr2score, open("%s%s.pik" % (directory, "nn2vr2score"), "w"))
  print "...done"

  print "combining..."
  # Choose how to combine the values.
  print "sample values:"
  for n, stuff in enumerate(nn2vr2score.iteritems()):
    print stuff
    if n > 5:
      break
  flatAvg = {nn: numpy.mean(vr2score.values()) for nn, vr2score in nn2vr2score.iteritems()} # Flat average, doesn't care about # of images.
  print "...done"

  outname = "%s%s.pik" % (directory, "flat_avg")
  print "saving flat averages to %s" % outname
  cPickle.dump(flatAvg, open(outname, "w"))

  # TODO delete this:

  """
  nn2vr2score has > 1 values.
  print str(nn2vr2score)
  for nn, vr2score in nn2vr2score.iteritems():
    for vr, score in vr2score.iteritems():
      if score > 1:
        print "score > 1 for %s" % str(nn)
  """
  return

  print "investigating distances..."
  for nn,v in flatAvg.iteritems():
    if v > 1:
      print "k=%s, v=%s" % (str(nn), str(v))
      print "===> %s" % str(nn2vr2score[nn])

  # TODO you'll probably want weighted averages, but start with this for now.

def twoFixedAvgStub():
  getAveragedRankings("data/twoFixedDist/")

def twoFixedGetHtmlStub():
  generateHTMLForExp("data/twoFixedDist/")

def generateHTMLForExp(loc):
  """
  TODO: Look at the top 5 and bottom 5 scoring classes. Print a few roles in which they're involved.
  For now, let's just choose some.
  """
  chosen = [("riding", "vehicle"), ("crawling", "agent"), ("distracting", "place"), ("jumping", "obstacle"), ("jumping", "agent"), ("jumping", "destination")]

  similarities = cPickle.load(open("%sflat_avg.pik" % loc, "r"))
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired

  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data) # TODO just save this.


  nn2vr2score = cPickle.load(open("%s%s.pik" % (loc, "nn2vr2score"), "r"))

  # Get a sorted list of the similarities, excluding duplicates
  similarities = [s for s in similarities if s[0] < s[1]]
  print "===> TOP 5"
  for n, similarity in enumerate(similarities[:5]):
    print "--> %d: %s" % (n, similarity)

  for n, similarity in enumerate(similarities, start=(len(similarities)-5)):
    print "--> %d: %s" % (n, similarity)

  showTopN = 5
  showBotN = 5

  goodToShow = set([vr for nn, vr in nn2vr2score.iteritems() if nn in similarities[:5]])
  badToShow = set([vr for nn, vr in nn2vr2score.iteritems() if nn in similarities[-5:]])
  print "Good Examples: %s" % goodToShow
  print "Bad Examples: %s" % badToShow
  toShow = goodToShow | badToShow

  for verb, role in toShow:
    print "Generating html for %s-%s" % (verb, role)
    generateHTMLHelper(similarities, verb, role, vrn2Imgs, loc)
#def getHTMLHelper(similarities, verb, role, vrn2Imgs):

def debugVerb(verb):
  # TODO use this to drill into a verb, if needed.
  # Find the probabilities for a particular verb
  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data) # TODO just save this.

  weights = {}
  weights[1] = collections.defaultdict(float)
  weights[1][1] = 1 # Whatever...
  weights[2] = collections.defaultdict(float)
  weights[2][2] = 1.
  weights[3] = collections.defaultdict(float)
  weights[3][2] = 1.
  weights[4] = collections.defaultdict(float)
  weights[4][2] = 1.
  weights[5] = collections.defaultdict(float)
  weights[5][2] = 1.
  weights[6] = collections.defaultdict(float)
  weights[6][2] = 1.
  weights[7] = collections.defaultdict(float)
  weights[7][2] = 1.

  myprob = lambda x,y,z: allgramProb(x,y,z,weights)

  vrProbs = VRProbDispatcher(data, [verb], myprob)

  nouns = getAllNounsForVR(verb, role, data)
  sim = getSimilarities(verb, role, nouns, vrProbs)

  for nn, v in sim.iteritems():
    if v > 1.:
      print nn,v

if __name__ == '__main__':
  print "running vecStyleStub"
  vecStyleStub()
