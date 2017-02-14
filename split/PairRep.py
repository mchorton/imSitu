import role_probabilities as rp
import utils as ut
import json
import data_utils as du

def sampleImg1(sample):
  return sample[3]
def sampleImg2(sample):
  return sample[4]

def getImg2NiceLabel(dataset):
  """
  return a dict like: {imgname: [{"role1": "label1"}, {"role1": "label2"}, ...]}
  where the labels are nice labels (not noun codes)
  """
  img2NiceLabel = {}
  for imgname, labeling in dataset.iteritems():
    verb = labeling["verb"]
    niceFrames = [{role: du.decodeNoun(noun) for role, noun in frame.iteritems()} for frame in labeling["frames"]]
    img2NiceLabel[imgname] = niceFrames
  return img2NiceLabel

"""
Representation of a set of image pairs. E.g., 
(noun1, noun2, similarities, img1name, img2name, vr) # TODO you also want the role in which the verbs differ.
"""
class PairRep(object):
  def __init__(self, datafileName):
    self.data = du.get_joint_set(datafileName)
    self.img2NiceLabels = getImg2NiceLabel(self.data)

    pass
  def load(self, filename):
    # We assume these are already sorted by similarities. It only really matters
    # for cosmetic reasons.
    self.samples = json.load(open(filename))

    # Get the 
  def getVRSamples(self, outHTML, vr):
    htmlTable = ut.HtmlTable()
    for sample in self.samples:
      if list(sample[-1]) == list(vr):
        nicelabel1 = self.img2NiceLabels[sampleImg1(sample)]
        nicelabel2 = self.img2NiceLabels[sampleImg2(sample)]
        htmlTable.addRowNice(set([rp.getImgUrl(sample[3])]), set([rp.getImgUrl(sample[4])]), "d(%s,%s)=%.4f, role=%s" % (du.decodeNoun(sample[0]), du.decodeNoun(sample[1]), sample[2], sample[-1]), str(nicelabel1), str(nicelabel2))
    with open(outHTML, "w") as f:
      f.write(str(htmlTable))

def get_vr_html(pairData = "data/vecStyle/chosen_pairs_thresh_2.0__freqthresh_10__bestNounOnly_True__blacklistprs_[set(['woman', 'man'])]__noThreeLabel_True.json", vrs = [("pushing", "item"), ("licking", "item"), ("rinsing", "object"), ("pouring", "substance")]):
  train = "zsTrain.json"
  rep = PairRep(train)
  rep.load(pairData)

  for vr in vrs:
    outHTML = pairData.replace(".json", "_vr_%s__.html" % str(vr))
    rep.getVRSamples(outHTML, vr)
