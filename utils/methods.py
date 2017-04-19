import os
import utils.mylogger as logging
import json
import itertools as it
def makeDirIfNeeded(filename):
  """
  Create the given directory, or the directory in which the file would be
  located.
  # filename: the name of the file or directory
  # if it's intended to be a directory, it should end with a slash!
  """
  directory = os.path.dirname(filename)
  if not os.path.exists(directory):
    os.makedirs(directory)

def loadJsonFile(filename):
    logging.getLogger(__name__).info("Reading '%s'" % filename)
    with open(filename) as myfile:
        return json.load(myfile)

def reverseMap(mydict):
    return {v : k for k, v in mydict.iteritems()}

def getDictLam(mydict):
    return lambda x: mydict[x]

def histString(histogram):
    """
    Represent a np.histogram in a nice / readable way.
    """
    hist, binEdges = histogram
    del histogram
    ret = "[" + "<%s>" % str(binEdges[0])
    binEdges = binEdges[1:]
    for val, edge in it.izip(hist, binEdges):
        ret += " %s <%s>" % (val, edge)
    return ret + ")"

