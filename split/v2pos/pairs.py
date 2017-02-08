"""
Generate pairs of images for the purpose of learning image transformations.
"""

import os
import re
import itertools as it
import htmlgen
import v2pos as v2pos

class FrameInfo(object):
  nameRegex = re.compile("([a-z]+)_season([a-z0-9]+)_disc(\d+)_(\d+)-(\d+).jpg")
  def __init__(self, filename):
    match = self.nameRegex.match(filename)
    if match is None:
      print "Couldn't match filename '%s' with regex for frame info."
      return
    self.show, self.season, self.disc, self.subdisc, self.frame = match.group(*range(1, 6))

    self.season = int(self.season)
    self.disc = int(self.disc)
    self.subdisc = int(self.subdisc)
    self.frame = int(self.frame)

  def comparator(self):
    return (self.show, self.season, self.disc, self.subdisc, self.frame)

  def __comp__(self, other):
    return cmp(self.comparator(), other.comparator())

  def __str__(self):
    return "%s_season%d_disc%d_%d-%08d.jpg" % (self.show, self.season, self.disc, self.subdisc, self.frame)

def get_image_pairs(filenames):
  imgsets = get_image_frames(filenames)
  return [(img1, img2) for imgset in imgsets for img1, img2 in it.combinations(imgset, 2)]

def get_image_frames(filenames):
  """
  Get a list of lists, each of which contains the images for a frame, properly
  sorted.
  """
  frameInfos = sorted([FrameInfo(filename) for filename in filenames], cmp=FrameInfo.__comp__)
  if len(frameInfos) == 0:
    print "No FrameInfos. Returning empty list!"
    return []
  examples = []
  curlist = [frameInfos[0]]
  for fi in frameInfos[1:]:
    expectedNext = list(curlist[-1].comparator())
    expectedNext[-1] += 2 # We expect the frame to advance by 2
    expectedNext = tuple(expectedNext)

    if expectedNext != fi.comparator():
      examples.append(curlist)
      curlist = []
    curlist.append(fi)
  examples.append(curlist)
  return [map(str, sublist) for sublist in examples]

def get_imgref(img):
  return htmlgen.imgref("annotated_images/" + img)


def generateHTML():
    # TODO you can use "annotated_images" instead.
    datadir = "data/VideoPose2/annotated_images/"
    outhtml = "data/VideoPose2/pairs.html"

    filenames = sorted([filename for filename in os.listdir(datadir) if filename.endswith(".jpg")])

    pairs = get_image_pairs(filenames)
    
    table = htmlgen.HtmlTable()

    for pair in pairs:
      table.addRow(map(get_imgref, pair))

    table.save(outhtml)

def viewClips():
  outhtml = "data/VideoPose2/clips.html"
  filenames = v2pos.getFileNames()
  frames = get_image_frames(filenames)
  table = htmlgen.HtmlTable()

  for n, clip in enumerate(frames, start=1):
    table.addRow(("Frame %d/%d" % (n, len(frames)),))
    table.addRow(map(get_imgref, clip))
  table.save(outhtml)
