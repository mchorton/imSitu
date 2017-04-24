import os
import utils.mylogger as logging
import json
import itertools as it
import sys
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

def setOutputToFiles(basename):
    stdoutfile = "%s.stdout" % basename
    stderrfile = "%s.stderr" % basename

    #logging.getLogger(__name__).info("Redirecting to files with base %s" % basename)

    sys.stdout = open(stdoutfile, 'w')
    sys.stderr = open(stderrfile, 'w')

    # Change the root logger to use the new stdout
    logging.reconfigure()

def pathinsert(pathname, inserted):
    """
    inserts "inserted" into pathname, just before the extension
    """
    pre, post = os.path.splitext(pathname)
    return "%s%s%s" % (pre, inserted, post)
