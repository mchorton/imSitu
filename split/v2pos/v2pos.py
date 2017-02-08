"""
Code for manipulating the v2pos data.
"""
import os
_DATADIR = "data/VideoPose2/annotated_images/"
def getFileNames():
  global _DATADIR
  datadir = _DATADIR
  filenames = sorted([filename for filename in os.listdir(datadir) if filename.endswith(".jpg")])
  return filenames
