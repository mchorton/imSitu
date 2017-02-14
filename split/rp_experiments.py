import role_probabilities as rp
import data_utils as du
import collections
import cPickle
import multiprocessing
import os

def saveAllDistances(data, vrProbs, outdir, nProc=None):
  v2r = du.getv2r(data)

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

def generateAllDistVecStyle(loc):
  dataset = ["zsTrain.json"]
  data = du.get_joint_set(dataset)

  vrn2Imgs = du.getvrn2Imgs(data)

  print "Getting Wholistic Sim Obj"
  wsc = rp.WholisticSimCalc(vrn2Imgs)

  vrProbs = rp.VRProbDispatcher(data, None, None, wsc.getWSLam()) # TODO want to also save the whole sim calc.
  saveAllDistances(data, vrProbs, loc)
  cPickle.dump(wsc, open("data/vecStyle/wsc.pik", "w"))

def generateW2VDist(dirName):
  modelFile = "data/word2vec/GoogleNews-vectors-negative300.bin"
  dataFileName = ["zsTrain.json"]
  print "Loading Data"
  data = du.get_joint_set(dataFileName)

  print "Getting vrn2Imgs"
  vrn2Imgs = du.getvrn2Imgs(data) # TODO make this cache or store as well!

  print "Loading Word2Vec"
  model = w2v.Word2Vec.load_word2vec_format(modelFile, binary=modelFile.endswith(".bin"))

  def getSim(vrprob, role, x, y):
    x,y = map(du.decodeNoun, (x,y))
    if x in model and y in model:
      return model.similarity(x,y)
    print "one of (x,y)=%s,%s not in model" % (str(x), str(y))
    return -9999999.

  print "Getting similarities"
  #getSim = lambda vrprob,role,x,y: model.similarity(x,y)
  vrProbs = rp.VRProbDispatcher(data, dfunc=getSim)
  rp.saveAllDistances(data, vrProbs, dirName)

def getAveragedRankings(directory):
  dataset = ["zsTrain.json"]
  data = du.get_joint_set(dataset)
  v2r = du.getv2r(data)
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

def generateHTMLHelper(similarities, verb, role, vrn2Imgs, outdir):
  """
  similarities - a list like [(noun1, noun2, similarity_score as a float), (...), ...]
  """
  htmlTable = ut.HtmlTable()
  dataset = sp.get_joint_dataset("zsTrain.json") # TODO a tad hacky.
  img2NiceLabel = getImg2NiceLabel(dataset)

  for k in similarities:
    img1Urls = getImgUrls(getNounImgs(verb, role, k[0], vrn2Imgs))
    img2Urls = getImgUrls(getNounImgs(verb, role, k[1], vrn2Imgs))
    if img1Urls == img2Urls or 0 in map(len, (img1Urls, img2Urls)):
      continue
    nicelabel1 = str(img2NiceLabels[k[3]])
    nicelabel2 = str(img2NiceLabels[k[4]])
    # TODO add the nice labels.
    htmlTable.addRowNice(img1Urls, img2Urls, "d(%s,%s)=%.4f" % (du.decodeNoun(k[0]), du.decodeNoun(k[1]), k[2]), nicelabel1, nicelabel2)
  with open(outdir + "%s-%s.html" % (verb, role), "w") as f:
    f.write(str(htmlTable))

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
  data = du.get_joint_set(dataset)

  vrn2Imgs = du.getvrn2Imgs(data) # TODO just save this.


  nn2vr2score = cPickle.load(open("%s%s.pik" % (loc, "nn2vr2score"), "r"))

  # Get a lookup set from noun to images for that noun
  n2Imgs = {}
  for vrn, images in vrn2Imgs.iteritems():
    noun = vrn[2]
    if vrn not in n2Imgs:
      n2Imgs[noun] = set()
    n2Imgs[noun] |= images

  # Get a sorted list of the similarities, excluding duplicates
  similarities = [s for s in similarities if s[0] < s[1]]
  # remove all noun pairs that have the same images
  similarities = [s for s in similarities if n2Imgs[s[0]] != n2Imgs[s[1]]]
  print "===> TOP 5"
  for n, similarity in enumerate(similarities[:5]):
    print "--> %d: %s" % (n, map(du.decodeNoun, similarity))

  for n, similarity in enumerate(similarities[-5:], start=(len(similarities)-5)):
    print "--> %d: %s" % (n, map(du.decodeNoun, similarity))

  showTopN = 5
  showBotN = 5

  print "nn2vrkeys: %s" % nn2vr2score.keys()[:5]

  #goodToShow = set([verbrole for verbrole in vr.keys() for nn, vr in nn2vr2score.iteritems() if nn in map(lambda x: (x[0],x[1]), similarities[:5])])
  #badToShow = set([verbrole for verbrole in vr.keys() for nn, vr in nn2vr2score.iteritems() if nn in map(lambda x: (x[0],x[1]), similarities[-5:])])
  goodToShow = set()
  badToShow = set()

  for nn, vr in nn2vr2score.iteritems():
    if nn in map(lambda x: (x[0], x[1]), similarities[:5]):
      goodToShow |= set(vr.keys())
    if nn in map(lambda x: (x[0], x[1]), similarities[-5:]):
      badToShow |= set(vr.keys())

  print "Good Examples: %s" % goodToShow
  print "Bad Examples: %s" % badToShow
  toShow = goodToShow | badToShow

  for verb, role in toShow:
    print "Generating html for %s-%s" % (verb, role)
    generateHTMLHelper(similarities, verb, role, vrn2Imgs, loc)
