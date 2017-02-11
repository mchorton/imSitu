import role_probabilities as rp
import json

"""
Representation of a set of image pairs. E.g., 
(noun1, noun2, similarities, img1name, img2name, vr) # TODO you also want the role in which the verbs differ.
"""
class PairRep(object):
  def __init__(self):
    pass
  def load(self, filename):
    # We assume these are already sorted by similarities. It only really matters
    # for cosmetic reasons.
    self.samples = json.load(open(filename))
  def getVRSamples(self, outHTML, vr):
    htmlTable = rp.HtmlTable()
    for sample in self.samples:
      if list(sample[-1]) == list(vr):
        htmlTable.addRow(set([rp.getImgUrl(sample[3])]), set([rp.getImgUrl(sample[4])]), "d(%s,%s)=%.4f, role=%s" % (rp.decodeNoun(sample[0]), rp.decodeNoun(sample[1]), sample[2], sample[-1]))
    with open(outHTML, "w") as f:
      f.write(str(htmlTable))

def get_vr_html(pairData = "data/vecStyle/chosen_pairs_thresh_2.0__freqthresh_10__bestNounOnly_True__blacklistprs_[set(['woman', 'man'])]__noThreeLabel_True.json", vrs = [("pushing", "item"), ("licking", "item"), ("rinsing", "object"), ("pouring", "substance")]):
  rep = PairRep()
  rep.load(pairData)

  for vr in vrs:
    outHTML = pairData.replace(".json", "_vr_%s__.html" % str(vr))
    rep.getVRSamples(outHTML, vr)
