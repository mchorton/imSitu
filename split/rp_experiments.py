import cPickle
import gensim.models.word2vec as w2v
import itertools as it
import json
import copy
import PairRep as pr
import splitutils as ut
import numpy as np
import role_probabilities as rp
import data_utils as du
import collections
import cPickle
import multiprocessing
import os
import utils.mylogger as logging
import tqdm
import split.v2pos.htmlgen as html
import splitutils as sut

vecStyleDirectory = "data/vecStyle"

def saveAllDistances(data, vrProbs, outdir, nProc=None, useCache=False):
  v2r = du.getv2r(data)

  needUpdate = False

  # TODO add multiprocessing here.
  for i, (verb,roles) in tqdm.tqdm(
      enumerate(v2r.iteritems()), total=len(v2r),
      desc="getting distance on per-verb basis"):
    outfile = "%s%s.pik" % (outdir, verb)
    if useCache and os.path.isfile(outfile):
      continue
    needUpdate = True
    vrn2dist = {role: vrProbs.getAllSimilarities(verb, role) for role in roles}
    cPickle.dump(vrn2dist, open(outfile, "w"))
  return needUpdate

def generateAllDistVecStyle(outdir, dataset):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  logging.getLogger(__name__).info("Getting dataset")
  data = du.get_joint_set(dataset)
  logging.getLogger(__name__).info("Getting vrn2Imgs")
  vrn2Imgs = du.getvrn2Imgs(data)

  logging.getLogger(__name__).info("Getting Wholistic Sim Obj")
  wsc = rp.WholisticSimCalc(vrn2Imgs)

  vrProbs = rp.VRProbDispatcher(data, None, None, wsc.getWSLam())
  needUpdate = saveAllDistances(data, vrProbs, outdir)
  if needUpdate:
    cPickle.dump(wsc, open(os.path.join(outdir, "wsc.pik"), "w"))

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
def getVrnData(*args, **kwargs):
  outDir = args[2]
  if not os.path.exists(outDir):
    os.makedirs(outDir)
  argsFile = os.path.join(outDir, "args.txt")
  with open(argsFile, "w") as argFile:
    argFile.write("ARGS=%s\nKWARGS=%s" % (str(args), str(kwargs)))
  return getVrnDataActual(*args, **kwargs)

def getVrnDataActual(distdir, datasetName, outDir, thresh=2, freqthresh = 10, blacklistprs = [set(["man", "woman"]), set(["man", "person"]), set(["woman", "person"]), set(["child", "male child"]), set(["child", "female child"])], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True):
  """
  gets a json object describing image pairs.
  pairing_score, image1, image2, transformation_role, image1_noun_value, image2_noun_value, image1_merged_reference, image2_merged_reference
  """

  toShow = rp.getSimilaritiesList(distdir, datasetName, thresh, freqthresh, blacklistprs, bestNounOnly, noThreeLabel, noOnlyOneRole, strictImgSep)

  # Need a DataExaminer to get the canonical representation for an image.
  dataset = du.get_joint_set(datasetName)
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

  outLoc = os.path.join(outDir, "vrnData.json")
  decodedLoc = os.path.join(outDir, "_decoded_vrnData.json")
  logging.getLogger(__name__).info("Writing vrndata to %s" % outLoc)
  json.dump(myobj, open(outLoc, "w+"))
  json.dump(myDecodedObj, open(decodedLoc, "w+"))

  makeMyHtml(outDir, myDecodedObj)

def makeMyHtml(outDir, decodedVrnData, maxElems = 500):
  logging.getLogger(__name__).info("Making HTML")

  # sort by score.
  sortedData = sorted(decodedVrnData, key=lambda x: x[0])

  stride = max(int(len(sortedData) / maxElems), 1)

  table = html.HtmlTable()
  table.addRow("Score", "Img1", "Img2", "Swapped Role", "Noun1", "Noun2", "Annotation1", "Annotation2")
  for elem in decodedVrnData[::stride]:
    # Change the image names into proper URLs
    elem[1] = html.ImgRef(src=sut.getUrl(elem[1]))
    elem[2] = html.ImgRef(src=sut.getUrl(elem[2]))
    table.addRow(*elem)

  argsFile = os.path.join(outDir, "args.txt")
  arguments = html.PhpTextFile(os.path.abspath(argsFile))

  maker = html.HtmlMaker()
  maker.addElement(arguments)
  maker.addElement(table)

  htmlOut = os.path.join(outDir, "index.php")
  logging.getLogger(__name__).info("Writing vrndata html to %s" % htmlOut)
  maker.save(htmlOut)

# TODO make the distance metric better.
def vrnDataStub():
  getVrnData("data/vecStyle/", thresh=2, freqthresh = 10, blacklistprs = [], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True)
