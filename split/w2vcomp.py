import cPickle
import role_probabilities as rp
import data_utils as sp
import gensim.models.word2vec as w2v
import itertools as it
import json
import copy
import collections
import PairRep as pr
import utils as ut
from nltk import wordnet as wn

"""
TODO move this.
"""
def makeAll():
  dirName = "data/word2vec/"
  #rp.generateW2VDist(dirName)
  #rp.getAveragedRankings(dirName)
  rp.generateHTMLForExp(dirName)

def get_wn_map():
  wn_map = {}
  for s in wn.wordnet.all_synsets():
    if s.pos() == 'n' :
      wn_map["n{0:08d}".format(s.offset())] = s;
  return wn_map

def get_best_noun(nouns, wn_map):
  bestnoun = None
  def getDists(n, nouns):
    nSyn = wn_map[n]
    totDist = 0
    for myn in nouns:
      if myn == "":
        continue
      totDist += nSyn.shortest_path_distance(wn_map[myn])
    return totDist

  if len(set(nouns)) != len(nouns):
    # There's a repeated element. Find it, and choose it.
    seen = set()
    for n in nouns:
      if n in seen:
        bestnoun = n
        break
      seen.add(n)
    if bestnoun is None:
      print "Invalid state!"
      exit(1)
  else:
    # There's no repeated element.
    d = {}
    for n in nouns:
      if n == "":
        continue
      d[n] = getDists(n, nouns)
    bestnoun = min(d, key=lambda key: d[key])
  if bestnoun is None:
    print "ERROR: bestnoun is None!"
    exit(1)
  return bestnoun

class DataExaminer(object):
  """
  Class used to preprocess the data and store useful information.
  """
  def __init__(self):
    pass
  def analyze(self, dataset):
    self.wn_map = get_wn_map()
    self.im2vr2nouns = get_im2vr2nouns(dataset)
    #self.im2vr2bestnoun = get_im2vr2bestnoun(dataset)
    #self.imgdeps = 
  def getBestNoun(self, image, vr):
    return get_best_noun(self.getNouns(image, vr), self.wn_map)
  def getNouns(self, image, vr):
    return self.im2vr2nouns[image][vr]

def get_im2vr2nouns(dataset):
  """
  Dataset: the loaded data.
  """
  wn_map = get_wn_map()
  ret = {}
  for imgname, labeling in dataset.iteritems():
    verb = labeling["verb"]
    vr2nouns = {}
    for frame in labeling["frames"]:
      for role, noun in frame.iteritems():
        vr2nouns[(verb, role)] = vr2nouns.get((verb, role), []) + [noun]
    ret[imgname] = vr2nouns
  return ret

def get_im2vr2bestnoun(dataset):
  # TODO I'd like to deprecate this, but it's way faster than the data examiner(?)
  """
  Dataset: the loaded data.
  """
  wn_map = get_wn_map()
  ret = {}
  for imgname, labeling in dataset.iteritems():
    verb = labeling["verb"]
    vr2nouns = {}
    for frame in labeling["frames"]:
      for role, noun in frame.iteritems():
        vr2nouns[(verb, role)] = vr2nouns.get((verb, role), []) + [noun]
    for vr, nouns in vr2nouns.iteritems():
      noun = get_best_noun(nouns, wn_map)
      vr2nouns[vr] = noun
    ret[imgname] = vr2nouns
  return ret

# TODO this functionality should be part of the ImgDep class.
def getim2vr2bestnoun(wn_map, imgdeps):
  exit(1)
  ret = {}
  for img,deps in imgdeps.iteritems():
    ret[img] = {}
    for role,nouns in deps.role2NounList.iteritems():
      print "role=%s" % str(role)
      print "nouns=%s" % str(nouns)
      if len(set(nouns)) != len(nouns):
        # There's a repeated element. Find it, and choose it.
        seen = set()
        for n in nouns:
          if n in seen:
            bestnoun = n
            break
          seen.add(n)
        print "Invalid state!"
        exit(1)

      else:
        # There's no repeated element.
        d = {}
        for n in nouns:
          if n == "":
            continue
          d[n] = getDists(n, nouns)
        if len(d) == 0:
          print "ERR:"
          print "nouns=%s" % str(nouns)
          print "role=%s" % str(role)
          print "imgs=%s" % str(img)
        bestnoun = max(d, key=lambda key: d[key])
      ret[img][role] = bestnoun
  return ret

# TODO un-blacklist "man, woman"
def makeHTML(dirName, thresh=2., freqthresh = 10, blacklistprs = [set(["man", "woman"])], bestNounOnly = True, noThreeLabel = True):
  """
  Make HTML that shows one image pair per line, ordered by distance between the
  images in similarity space.
  freqthresh: if a noun occurs freqthresh or fewer times, it'll be excluded.
  blacklistprs: each pair (n1, n2) that matches 
  """
  loc = dirName
  datasets = ["zsTrain.json"]
  #datasets = ["zsSmall.json"]
  similarities = cPickle.load(open("%sflat_avg.pik" % loc, "r"))
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired

  train = du.get_joint_set(datasets)
  vrn2Imgs = du.getvrn2Imgs(train)

  n2vr2Imgs = {}
  for vrn, imgs in vrn2Imgs.iteritems():
    v,r,n = vrn
    if n not in n2vr2Imgs:
      n2vr2Imgs[n] = {}
    if (v,r) not in n2vr2Imgs[n]:
      n2vr2Imgs[n][(v,r)] = set()
    n2vr2Imgs[n][(v,r)] |= imgs

  #toShow = {} # Map from (n1,n2,sim) -> set([(img1, img2), ...])
  toShow2 = [] # list like (n1,n2,sim,img1,img2)

  print "Total num sims: %d" % len(similarities)

  imgdeps = du.getImageDeps(vrn2Imgs)

  # Get the number of images in which each noun occurs.
  freqs = du.getFrequencies(imgdeps)

  # Precache wordnet distances
  wn_map = get_wn_map()

  im2vr2bestnoun = get_im2vr2bestnoun(train)
  examiner = DataExaminer()
  examiner.analyze(train)

  stats = collections.defaultdict(int)
  nPassSame = 0
  nPassDiff = 0
  nNotBestLabel = 0 # This number is meaningless since I didn't check roles before incrementing. Oh well.
  print "looping..."
  for n1, n2, sim in similarities:
    if n1 == "" or n2 == "":
      continue
    if set([rp.decodeNoun(n1),rp.decodeNoun(n2)]) in blacklistprs:
      continue
    if freqs[n1] < freqthresh or freqs[n2] < freqthresh:
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
        dl = imgdeps[one].differentLabelings(imgdeps[two])
        if len(dl) == 1:
          if dl[0] == vr:
            nPassSame += 1
            toShow2.append([n1, n2, sim, one, two, tuple(vr)])
          else:
            nPassDiff += 1
  
  print "nPassSame=%s" % str(nPassSame)
  print "nPassDiff=%s" % str(nPassDiff)
  print "nNotBestLabel=%s" % str(nNotBestLabel)
  print "stats=%s" % str(stats)

  print "There are %d valid pairs" % len(toShow2)
  toShow2 = [t for t in toShow2 if t[2] < thresh]
  print "Cutoff thresh %f: there are %d valid pairs" % (thresh, len(toShow2))
  print "Expanding (n1, n2) => (n1, n2), (n2, n1)"
  nextShow = []
  # Generate reverse pairs.
  for elem in toShow2:
    nextShow.append(elem)
    newelem = copy.deepcopy(elem)
    newelem[0] = elem[1]
    newelem[1] = elem[0]
    newelem[3] = elem[4]
    newelem[4] = elem[3]
    nextShow.append(newelem)
  toShow2 = nextShow
  print "There are now %d valid pairs" % len(toShow2)
  suffix = "%s_%s__%s_%s__%s_%s__%s_%s__%s_%s" % ("thresh", str(thresh), "freqthresh", str(freqthresh), "bestNounOnly", str(bestNounOnly), "blacklistprs", str(blacklistprs), "noThreeLabel", str(noThreeLabel))
  json.dump(toShow2, open(dirName + "chosen_pairs_%s.json" % (suffix), "w"))

  subsampleFactor = 1000
  print "Subsampling similarities by %d" % subsampleFactor
  stride = len(toShow2) / subsampleFactor
  cont = toShow2[:20]
  if stride > 0:
    toShow2 = cont + toShow2[::stride]

  # TODO rest of this should be a separate function
  img2NiceLabels = pr.getImg2NiceLabel(train)

  # get a table
  htmlTable = ut.HtmlTable()
  for n1, n2, sim, one, two, vr in toShow2:
    nicelabel1 = str(img2NiceLabels[one])
    nicelabel2 = str(img2NiceLabels[two])
    img1urls = set([rp.getImgUrl(one)])
    img2urls = set([rp.getImgUrl(two)])
    htmlTable.addRowNice(img1urls, img2urls, "d(%s,%s)=%.4f, imgs=(%s, %s, %s)" % (rp.decodeNoun(n1), rp.decodeNoun(n2), sim, one, two, str(vr)), nicelabel1, nicelabel2)

  with open(dirName + "all_sim_prs_%s.html" % (suffix), "w") as f:
    f.write(str(htmlTable))
