import role_probabilities as rp
import unittest
import math

class TestRP(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass
  def test_vrp(self):
    frames = [{"role1": "noun1", "role2": "noun2"}, {"role1": "noun1", "role2": "noun2b"}]
    vrp = rp.VRProb()
    vrp.add(frames)

    self.assertEqual(("role1", "role2"), vrp.order)
    dist = vrp.getDistance("role1", "noun1", "noun1")
    self.assertEqual(dist, 0)

    dist = vrp.getDistance("role2", "noun2", "noun2")
    self.assertEqual(dist, 0)
    dist = vrp.getDistance("role2", "noun2", "noun2b")
    self.assertEqual(dist, 0)

  def test_vrp_more(self):
    frames = [
        {"role1": "noun1", "role2": "noun2"},
        {"role1": "noun1b", "role2": "noun2b"}
        ]
    vrp = rp.VRProb()
    vrp.add(frames)

    dist = vrp.getDistance("role1", "noun1", "noun1b")
    self.assertEqual(dist, 1)

  """
  # TODO I just assume this works.
  def test_vrp_even_more(self):
    frames = [
        {"role1": "noun1", "role2": "noun2"},
        {"role1": "noun1b", "role2": "noun2"},
        {"role1": "noun1b", "role2": "noun2b"},
        {"role1": "noun1b", "role2": "noun2b"}
        ]
    vrp = rp.VRProb()
    vrp.add(frames)

    dist = vrp.getDistance("role1", "noun1", "noun1b")
    self.assertEqual(dist, 0)
  """
