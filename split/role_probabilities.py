import split as split
import collections
import math
import json

class VRProb():
  def __init__(self):
    self.counts = None
    self.order = None # This is the order that the counts map's keys are in.
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
  def match(self, test, target, roleIndex): # TODO unused.
    # Determine whether 'test' matches 'target' everywhere except at roleindex
    return all([test[i] == target[i] for i in range(len(test)) if i != roleIndex])
  def getDistance(self, role, noun1, noun2):
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
      prob1 = 1. * self.counts[tuple(key)] / normalization1
      key[roleIndex] = noun2
      prob2 = 1. * self.counts[tuple(key)] / normalization2
      total += (math.sqrt(prob1) - math.sqrt(prob2)) ** 2
    return math.sqrt(total) / math.sqrt(2)

class VRProbDispatcher():
  def __init__(self, data, chosen = None):
    self.data = data
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
        self.verbVRPMap[verb] = VRProb()
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
  return _nouncache[noun]["gloss"][0]

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
    </style>
    </head>
    <body>
    """
    self.footer = """</body>
    </html>"""
  def addRow(self, urls1, urls2, label):
    self.rows.append([urls1, urls2, label])
  def __str__(self):
    output = ""
    output += self.header
    output += "<table>"
    for row in self.rows:
      output += "<tr>\n"
      for elem in row:
        output += "<th>%s<th>\n" % elem # TODO need to think about this.
        #output += "<th>dolphin</th>\n" # TODO need to think about this.
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
  return "".join(list(ret))
  #return ret

def getNounImgs(verb, role, noun, vrn2Images):
  return vrn2Images[(verb, role, noun)]

def generateHTML(verb, role, vrProbs, vrn2Imgs):
  outdir = "data/"
  chosen = [verb]
  similarities = vrProbs.getAllSimilarities(verb, role)
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired

  htmlTable = HtmlTable()

  for k in similarities:
    img1Urls = getImgUrls(getNounImgs(verb, role, k[0], vrn2Imgs))
    img2Urls = getImgUrls(getNounImgs(verb, role, k[1], vrn2Imgs))
    htmlTable.addRow(img1Urls, img2Urls, "d(%s,%s)=%.4f" % (decodeNoun(k[0]), decodeNoun(k[1]), k[2]))
  with open(outdir + "%s-%s.html" % (verb, role), "w") as f:
    f.write(str(htmlTable))

# TODO: jumping-agent (which has an 'empty-set' sign) has some bug...
def generateAllHTML():
  chosen = [("riding", "vehicle"), ("crawling", "agent"), ("distracting", "place"), ("jumping", "obstacle"), ("jumping", "agent"), ("jumping", "destination")]

  dataset = ["zsTrain.json"]
  data = split.get_joint_set(dataset)

  vrn2Imgs = split.getvrn2Imgs(data)
  vrProbs = VRProbDispatcher(data, chosen)

  for verb, role in chosen:
    print "Generating html for %s-%s" % (verb, role)
    generateHTML(verb, role, vrProbs, vrn2Imgs)
