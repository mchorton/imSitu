import utils as ut
import cPickle
import gensim.models.word2vec as w2v
import itertools as it
import json
import copy
import PairRep as pr
import utils as ut
import numpy as np
import role_probabilities as rp
import data_utils as du
import collections
import cPickle
import multiprocessing
import os

vecStyleDirectory = "data/vecStyle"

def saveAllDistances(data, vrProbs, outdir, nProc=None):
  v2r = du.getv2r(data)

  needUpdate = False

  for i, (verb,roles) in enumerate(v2r.iteritems()):
    print "Considering verb %d/%d" % ((i + 1), len(v2r))
    outfile = "%s%s.pik" % (outdir, verb)
    if os.path.isfile(outfile):
      print "...Found precached results for verb '%s'" % (verb)
      continue
    needUpdate = True
    print "...Computing all distances for verb '%s'..." % verb
    vrn2dist = {role: vrProbs.getAllSimilarities(verb, role) for role in roles}
    print "...Saving distances for verb '%s' to '%s'..." % (verb, outfile)
    cPickle.dump(vrn2dist, open(outfile, "w"))
    print "...done."
  return needUpdate

def generateAllDistVecStyle(loc):
  dataset = ["zsTrain.json"]
  data = du.get_joint_set(dataset)

  vrn2Imgs = du.getvrn2Imgs(data)

  print "Getting Wholistic Sim Obj"
  wsc = rp.WholisticSimCalc(vrn2Imgs)

  vrProbs = rp.VRProbDispatcher(data, None, None, wsc.getWSLam()) # TODO want to also save the whole sim calc.
  needUpdate = saveAllDistances(data, vrProbs, loc)
  # TODO this object doesn't have anything in it.
  # Maybe the lambda is shielding it? But I doubt that.
  if needUpdate:
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
  saveAllDistances(data, vrProbs, dirName)

def makeHTML(dirName, thresh=2., freqthresh = 10, blacklistprs = [set(["man", "woman"]), set(["man", "person"]), set(["woman", "person"]), set(["child", "male child"]), set(["child", "female child"])], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnylOneRole = True, strictImgSep=True, maxDbgLen = 1000):
  train = du.get_joint_set("zsTrain.json")

  cacher = ut.Cacher(dirName)
  print "Loading similarities..."

  suffix = "%s__%s__%s__%s__%s__%s__%s" % (str(thresh), str(freqthresh), str(bestNounOnly), str(blacklistprs), str(noThreeLabel), str(strictImgSep), str(noOnlyOneRole))
  #suffix = "%s_%s__%s_%s__%s_%s__%s_%s__%s_%s__%s_%s__%s_%s" % ("thresh", str(thresh), "freqthresh", str(freqthresh), "bestNounOnly", str(bestNounOnly), "blacklistprs", str(blacklistprs), "noThreeLabel", str(noThreeLabel), "strictImgSep", str(strictImgSep), "noOnlyOneRole", str(noOnlyOneRole))
  simName = dirName + "chosen_pairs_%s.json" % (suffix)

  #calculator = rp.SimilaritiesListCalculator(dirName, thresh=thresh, freqthresh=freqthresh, blacklistprs=blacklistprs, bestNounOnly=bestNounOnly, noThreeLabel=noThreeLabel)

  toShow = cacher.run(rp.getSimilaritiesList, dirName, thresh, freqthresh, blacklistprs, bestNounOnly, noThreeLabel, noOnlyOneRole, strictImgSep)
  #toShow = calculator.getSimilaritiesList()
  #toShow = rp.getSimilaritiesList(dirName, thresh, freqthresh, blacklistprs, bestNounOnly, noThreeLabel)

  print "There are now %d valid pairs" % len(toShow)
  #json.dump(toShow, open(dirName + "chosen_pairs_%s.json" % (suffix), "w"))

  subsampleFactor = 1000
  print "Subsampling similarities by %d" % subsampleFactor
  stride = len(toShow) / subsampleFactor
  cont = toShow[:20]
  if stride > 0:
    toShow = cont + toShow[::stride]

  # TODO rest of this should be a separate function
  img2NiceLabels = pr.getImg2NiceLabel(train)

  # Load the wsc if needed
  wsc = None
  if includeWSC:
    wsc = cPickle.load(open(dirName + "wsc.pik"))

  # get a table
  htmlTable = ut.HtmlTable()
  for n1, n2, sim, one, two, vr in toShow:
    nicelabel1 = str(img2NiceLabels[one])
    nicelabel2 = str(img2NiceLabels[two])
    img1urls = set([ut.getImgUrl(one)])
    img2urls = set([ut.getImgUrl(two)])
    dbg = ""
    if wsc:
      dbg = str(wsc.simDbg.get((n1, n2), ""))
      if dbg == "":
        dbg = str(wsc.simDbg.get((n2, n1), ""))
    if len(dbg) > maxDbgLen:
      msg = "... etc. (truncating examples due to length)"
      dbg = dbg[:maxDbgLen-len(msg)] + msg
    htmlTable.addRowNice(img1urls, img2urls, "d(%s,%s)=%.4f, imgs=(%s, %s, %s)" % (du.decodeNoun(n1), du.decodeNoun(n2), sim, one, two, str(vr)), nicelabel1, nicelabel2, dbg)

  with open(dirName + "all_sim_prs_%s.html" % (suffix), "w") as f:
    f.write(str(htmlTable))

#def makeHTML(dirName, thresh=2., freqthresh = 10, blacklistprs = [set(["man", "woman"]), set(["man", "person"]), set(["woman", "person"]), set(["child", "male child"]), set(["child", "female child"])], bestNounOnly = True, noThreeLabel = True, includeWSC=True, measureImgDistWithBest=True, maxDbgLen = 1000):

# Produce the vrnData used for later experiments.
def getJsonSummary(dirName, thresh=2, freqthresh = 10, blacklistprs = [set(["man", "woman"]), set(["man", "person"]), set(["woman", "person"]), set(["child", "male child"]), set(["child", "female child"])], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True):

  # gets a json object describing image pairs.
  # pairing_score, image1, image2, transformation_role, image1_noun_value, image2_noun_value, image1_merged_reference, image2_merged_reference

  cacher = ut.Cacher(dirName)
  toShow = cacher.run(rp.getSimilaritiesList, dirName, thresh, freqthresh, blacklistprs, bestNounOnly, noThreeLabel, noOnlyOneRole, strictImgSep)

  # Need a DataExaminer to get the canonical representation for an image.
  dataset = du.get_joint_set("zsTrain.json")
  examiner = du.DataExaminer()
  examiner.analyze(dataset)

  def stringifyKeys(obj):
    # return a copy of obj except the keys are lists instead of 
    return {str(k): v for k,v in obj.iteritems()}

  def decodeVals(obj):
    return {k: du.decodeNoun(v) for k,v in obj.iteritems()}

  myobj = []
  for stuff in toShow:
    myobj.append([stuff[2], stuff[3], stuff[4], list(stuff[5]), stuff[0], stuff[1], stringifyKeys(examiner.getCanonicalLabels(stuff[3])), stringifyKeys(examiner.getCanonicalLabels(stuff[4]))])

  myDecodedObj = []
  for stuff in toShow:
    myDecodedObj.append([stuff[2], stuff[3], stuff[4], list(stuff[5]), du.decodeNoun(stuff[0]), du.decodeNoun(stuff[1]), decodeVals(stringifyKeys(examiner.getCanonicalLabels(stuff[3]))), decodeVals(stringifyKeys(examiner.getCanonicalLabels(stuff[4])))])
  #suffix = "%s_%s__%s_%s__%s_%s__%s_%s__%s_%s__%s_%s__%s_%s" % ("thresh", str(thresh), "freqthresh", str(freqthresh), "bestNounOnly", str(bestNounOnly), "blacklistprs", str(blacklistprs), "noThreeLabel", str(noThreeLabel), "strictImgSep", str(strictImgSep), "noOnlyOneRole", str(noOnlyOneRole))
  suffix = "%s__%s__%s__%s__%s__%s__%s" % (str(thresh), str(freqthresh), str(bestNounOnly), str(blacklistprs), str(noThreeLabel), str(strictImgSep), str(noOnlyOneRole))
  #return myobj, myDecodedObj
  json.dump(myobj, open(dirName + "json_summary_%s.json" % str(suffix), "w+"))
  json.dump(myDecodedObj, open(dirName + "json_summary_decoded_%s.json" % str(suffix), "w+"))

# TODO make the distance metric better.
def summaryStub():
  getJsonSummary("data/vecStyle/", thresh=2, freqthresh = 10, blacklistprs = [], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True)
