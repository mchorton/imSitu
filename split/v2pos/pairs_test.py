import unittest
import split.v2pos.pairs as pairs

class TestV2Pos(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.maxDiff = None
    #cls.datafile = "testfunc.json"
    #cls.dataset = json.load(open(cls.datafile))
    #cls.vrn2Imgs = sp.getvrn2Imgs(cls.dataset)
    pass

  def test_FrameInfo(self):
    imgname = "friends_season1_disc1_3-00029474.jpg"
    info = pairs.FrameInfo(imgname)
    self.assertEqual(imgname, str(info))
    self.assertEqual(("friends", 1, 1, 3, 29474), info.comparator())

  def test_image_frames(self):
    imgnames = [
      "friends_season2_disc1_3-00029490.jpg",
      "friends_season1_disc1_3-00029476.jpg",
      "friends_season1_disc1_3-00029478.jpg",
      "friends_season1_disc1_3-00029480.jpg",
      "friends_season1_disc1_3-00029474.jpg",
      "friends_season2_disc1_3-00029482.jpg",
      "friends_season2_disc1_3-00029484.jpg",
      "friends_season2_disc1_3-00029488.jpg",
      ]
    expected = [
      [
      "friends_season1_disc1_3-00029474.jpg",
      "friends_season1_disc1_3-00029476.jpg",
      "friends_season1_disc1_3-00029478.jpg",
      "friends_season1_disc1_3-00029480.jpg",
      ],
      [
      "friends_season2_disc1_3-00029482.jpg",
      "friends_season2_disc1_3-00029484.jpg",
      ],
      [
      "friends_season2_disc1_3-00029488.jpg",
      "friends_season2_disc1_3-00029490.jpg",
      ]]
    self.assertEqual(expected, pairs.get_image_frames(imgnames))

  def test_image_pairs(self):
    imgnames = [
      "friends_season1_disc1_3-00029476.jpg",
      "friends_season1_disc1_3-00029478.jpg",
      "friends_season1_disc1_3-00029480.jpg",
      "friends_season2_disc1_3-00000002.jpg",
      "friends_season2_disc1_3-00000004.jpg"
      ]
    expected = [
      ("friends_season1_disc1_3-00029476.jpg", "friends_season1_disc1_3-00029478.jpg"),
      ("friends_season1_disc1_3-00029476.jpg", "friends_season1_disc1_3-00029480.jpg"),
      ("friends_season1_disc1_3-00029478.jpg", "friends_season1_disc1_3-00029480.jpg"),
      ("friends_season2_disc1_3-00000002.jpg", "friends_season2_disc1_3-00000004.jpg")
      ]
    self.assertEqual(sorted(expected), sorted(pairs.get_image_pairs(imgnames)))
