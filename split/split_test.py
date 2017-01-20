import unittest
import json
import split as sp
import collections

class TestSplit(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.datafile = "testfunc.json"
    cls.dataset = json.load(open(cls.datafile))
    cls.vrn2Imgs = sp.getvrn2Imgs(cls.dataset)
    pass
  def test_getvrn2Imgs(self):
    expected = {(u'verb2', u'place', u'p2'): set([u'img2a.jpg', u'img2.jpg']), (u'verb2', u'agent', u'a2'): set([u'img2a.jpg', u'img2.jpg']), (u'verb3', u'agent', u'a3'): set([u'img3b.jpg', u'img3.jpg']), (u'verb3', u'agent', u'p3'): set([u'img3.jpg', u'img3c.jpg']), (u'verb3', u'place', u'000'): set([u'img3b.jpg']), (u'verb3', u'place', u'p3'): set([u'img3.jpg']), (u'verb1', u'place', u'p1'): set([u'img1.jpg']), (u'verb1', u'agent', u'a1'): set([u'img1.jpg']), (u'verb3', u'place', u'0'): set([u'img3c.jpg'])}
    self.assertEqual(self.vrn2Imgs, expected)

  def test_getImageDeps(self):
    deps = sp.getImageDeps(self.vrn2Imgs)
    obs = str(deps)

    expected = "{u'img2a.jpg': (img2a.jpg, {(u'verb2', u'agent'): set([u'img2a.jpg', u'img2.jpg']), (u'verb2', u'place'): set([u'img2a.jpg', u'img2.jpg'])}), u'img3b.jpg': (img3b.jpg, {(u'verb3', u'place'): set([u'img3b.jpg']), (u'verb3', u'agent'): set([u'img3b.jpg', u'img3.jpg'])}), u'img2.jpg': (img2.jpg, {(u'verb2', u'agent'): set([u'img2a.jpg', u'img2.jpg']), (u'verb2', u'place'): set([u'img2a.jpg', u'img2.jpg'])}), u'img1.jpg': (img1.jpg, {(u'verb1', u'agent'): set([u'img1.jpg']), (u'verb1', u'place'): set([u'img1.jpg'])}), u'img3.jpg': (img3.jpg, {(u'verb3', u'place'): set([u'img3.jpg']), (u'verb3', u'agent'): set([u'img3b.jpg', u'img3.jpg', u'img3c.jpg'])}), u'img3c.jpg': (img3c.jpg, {(u'verb3', u'place'): set([u'img3c.jpg']), (u'verb3', u'agent'): set([u'img3.jpg', u'img3c.jpg'])})}"

    self.assertEqual(obs, expected)

  def test_get_split(self):
    train, dev, _ = sp.get_split(self.datafile, 0.)
    self.assertEqual(len(dev), 1)


    train, dev, _ = sp.get_split(self.datafile, .2)
    self.assertEqual(len(dev), 2)

    train, dev, _ = sp.get_split(self.datafile, .4)
    self.assertEqual(len(dev), 3)

    train, dev, _ = sp.get_split(self.datafile, .6)
    self.assertEqual(len(dev), 4)

    train, dev, _ = sp.get_split(self.datafile, .8) # Last 2 will be moved together.
    self.assertEqual(len(dev), 6)

  def test_minNeededForZS(self):
    deps = sp.getImageDeps(self.vrn2Imgs)
    img1deps = deps["img1.jpg"]
    observed = img1deps.minNeededForZS(deps)
    expected = set(["img1.jpg"])
    self.assertEqual(observed, expected)

    img2deps = deps["img2.jpg"]
    observed = img2deps.minNeededForZS(deps)
    expected = set(["img2.jpg", "img2a.jpg"])
    self.assertEqual(observed, expected)

    img3cdeps = deps["img3c.jpg"]
    observed = img3cdeps.minNeededForZS(deps)
    expected = set(["img3c.jpg"])
    self.assertEqual(observed, expected)

    img3deps = deps["img3.jpg"]
    observed = img3deps.minNeededForZS(deps)
    expected = set(["img3.jpg"])
    self.assertEqual(observed, expected)

  def test_numDifferentLabelings(self):
    self.datafile = "testdiffer.json"
    counts = sp.getDifferers(self.datafile)
    expected = collections.defaultdict(int)
    expected[0] = 2
    expected[1] = 3
    expected[2] = 1
    self.assertEqual(counts, expected)

    self.datafile = "testfunc.json"
    counts = sp.getDifferers(self.datafile)
    expected[0] = 1
    expected[1] = 2
    expected[2] = 1
    self.assertEqual(counts, expected)

    self.datafile = ["testdiffer.json", "testfunc.json"]
    counts = sp.getDifferers(self.datafile)
    expected[0] = 3
    expected[1] = 5
    expected[2] = 2
    self.assertEqual(counts, expected)
    #self.assertEqual(True, False)

