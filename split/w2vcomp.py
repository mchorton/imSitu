import cPickle
import role_probabilities as rp
import split as sp
import gensim.models.word2vec as w2v
import itertools as it
import json
import copy

def generateW2VDist(dirName):
  modelFile = "data/word2vec/GoogleNews-vectors-negative300.bin"
  dataFileName = ["zsTrain.json"]
  print "Loading Data"
  data = sp.get_joint_set(dataFileName)

  print "Getting vrn2Imgs"
  vrn2Imgs = sp.getvrn2Imgs(data) # TODO make this cache or store as well!

  print "Loading Word2Vec"
  model = w2v.Word2Vec.load_word2vec_format(modelFile, binary=modelFile.endswith(".bin"))

  def getSim(vrprob, role, x, y):
    x,y = map(rp.decodeNoun, (x,y))
    if x in model and y in model:
      return model.similarity(x,y)
    print "one of (x,y)=%s,%s not in model" % (str(x), str(y))
    return -9999999.

  print "Getting similarities"
  #getSim = lambda vrprob,role,x,y: model.similarity(x,y)
  vrProbs = rp.VRProbDispatcher(data, dfunc=getSim)
  rp.saveAllDistances(data, vrProbs, dirName)

def makeAll():
  dirName = "data/word2vec/"
  #generateW2VDist(dirName)
  #rp.getAveragedRankings(dirName)
  rp.generateHTMLForExp(dirName)

def makeHTML(dirName, thresh=2., freqthresh = 0):
  """
  Make HTML that shows one image pair per line, ordered by distance between the
  images in similarity space.
  freqthresh: if a noun occurs freqthresh or fewer times, it'll be excluded.
  """
  loc = dirName
  datasets = ["zsTrain.json"]
  similarities = cPickle.load(open("%sflat_avg.pik" % loc, "r"))
  desired = [(k[0], k[1], v) for k,v in similarities.iteritems()]
  desired.sort(key=lambda x: x[2])
  similarities = desired


  train = sp.get_joint_set(datasets)
  vrn2Imgs = sp.getvrn2Imgs(train)

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

  # TODO n2Imgs thing?
  #similarities = [s for s in similarities if s[0] < s[1]]
  #similarities = [s for s in similarities if s[2] < 99]
  #similarities = similarities[:500] + similarities[-500:]
  #print "subsampling similarities..."
  #stride = len(similarities) / 50
  #similarities = similarities[::stride]

  imgdeps = sp.getImageDeps(vrn2Imgs)

  # Get the number of images in which each noun occurs.
  freqs = sp.getFrequencies(imgdeps)

  # TODO:
  # Should I expand to include "(n1, n2)" and "(n2, n1)"? Only includes one of them for now.
  # Should I remove rare nouns? I haven't yet.
  nPassSame = 0
  nPassDiff = 0
  for n1, n2, sim in similarities:
    if n1 == "" or n2 == "":
      continue
    if freqs[n1] < freqthresh or freqs[n2] < freqthresh:
      continue
    for vr, imgs in n2vr2Imgs[n1].iteritems():
      firstset = imgs
      secondset = n2vr2Imgs[n2].get(vr, [])
      for one,two in it.product(firstset, secondset):
        dl = imgdeps[one].differentLabelings(imgdeps[two])
        if len(dl) == 1:
          if dl[0] == vr:
            nPassSame += 1
            toShow2.append([n1, n2, sim, one, two])
          else:
            nPassDiff += 1
  
  print "nPassSame=%s" % str(nPassSame)
  print "nPassDiff=%s" % str(nPassDiff)


  print "There are %d valid pairs" % len(toShow2)
  toShow2 = [t for t in toShow2 if t[2] < thresh]
  print "Cutoff thresh %f: there are %d valid pairs" % (thresh, len(toShow2))
  print "Expanding (n1, n2) => (n1, n2), (n2, n1)"
  nextShow = []
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
  json.dump(toShow2, open(dirName + "chosen_pairs_thresh_%.3f_freqthresh_%d.json" % (thresh, freqthresh), "w"))

  subsampleFactor = 1000
  print "Subsampling similarities by %d" % subsampleFactor
  stride = len(toShow2) / subsampleFactor
  cont = toShow2[:20]
  toShow2 = cont + toShow2[::stride]

  # get a table
  htmlTable = rp.HtmlTable()
  for n1, n2, sim, one, two in toShow2:
    img1urls = set([rp.getImgUrl(one)])
    img2urls = set([rp.getImgUrl(two)])
    htmlTable.addRow(img1urls, img2urls, "d(%s,%s)=%.4f, imgs=(%s, %s)" % (rp.decodeNoun(n1), rp.decodeNoun(n2), sim, one, two))

  with open(dirName + "all_sim_prs_thresh_%.3f_freqthresh_%d.html" % (thresh, freqthresh), "w") as f:
    f.write(str(htmlTable))

if __name__ == '__main__':
  makeAll()
