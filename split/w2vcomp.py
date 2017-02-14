import cPickle
import data_utils as du
import role_probabilities as rp
import data_utils as sp
import gensim.models.word2vec as w2v
import itertools as it
import json
import copy
import collections
import PairRep as pr
import utils as ut

"""
TODO move this.
"""
def makeAll():
  dirName = "data/word2vec/"
  #rp.generateW2VDist(dirName)
  #rp.getAveragedRankings(dirName)
  rp.generateHTMLForExp(dirName)

# TODO un-blacklist "man, woman"
# TODO move this to another file!
def makeHTML(dirName, thresh=2., freqthresh = 10, blacklistprs = [set(["man", "woman"])], bestNounOnly = True, noThreeLabel = True, includeWSC=True, measureImgDistWithBest=True):
  train = du.get_joint_set("zsTrain.json")

  print "Loading similarities..."

  suffix = "%s_%s__%s_%s__%s_%s__%s_%s__%s_%s__%s_%s" % ("thresh", str(thresh), "freqthresh", str(freqthresh), "bestNounOnly", str(bestNounOnly), "blacklistprs", str(blacklistprs), "noThreeLabel", str(noThreeLabel), "midwb", str(measureImgDistWithBest))
  simName = dirName + "chosen_pairs_%s.json" % (suffix)
  calculator = rp.SimilaritiesListCalculator(dirName, thresh=thresh, freqthresh=freqthresh, blacklistprs=blacklistprs, bestNounOnly=bestNounOnly, noThreeLabel=noThreeLabel)

  toShow = calculator.getSimilaritiesList()
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
    htmlTable.addRowNice(img1urls, img2urls, "d(%s,%s)=%.4f, imgs=(%s, %s, %s)" % (du.decodeNoun(n1), du.decodeNoun(n2), sim, one, two, str(vr)), nicelabel1, nicelabel2, dbg)

  with open(dirName + "all_sim_prs_%s.html" % (suffix), "w") as f:
    f.write(str(htmlTable))
