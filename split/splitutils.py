import operator as op
import itertools
import os
import cPickle
import random

class HtmlTable():
  def __init__(self):
    self.rows = []
    self.header = """ <!DOCTYPE html>
    <html>
    <head>
    <style>
    table, th, td {
          border: 1px solid black;
    }
    td { white-space:pre }
    </style>
    </head>
    <body>
    """
    self.footer = """</body>
    </html>"""
    self.imPerRow = 4
  def addRow(self, urls1, urls2, label):
    # TODO this is deprecated
    self.rows.append([urls1, urls2, label])
  def addRowNice(self, *args):
    self.rows.append(args)
  def __str__(self):
    # TODO this is deprecated.
    output = ""
    output += self.header
    output += "<table>"
    for row in self.rows:
      output += "<tr>\n"
      for n, elem in enumerate(row):
        output += "<th>"
        if not isinstance(elem, basestring):
          output += "\n".join(map("".join, map(mytostr, *[iter(elem)]*self.imPerRow)))
        else:
          output += "%s" % elem
        output += "</th>\n" # TODO end th?
      output += "</tr>\n"
    output += "</table>"
    output += self.footer
    return output

def mytostr(*args):
  return tuple(["" if arg is None else arg for arg in args])

def getImgUrl(name):
  return '<img src="%s">' % (getUrl(name))

def getUrl(name):
  return 'https://s3.amazonaws.com/my89-frame-annotation/public/images_256/%s' % (name)

def getImgUrls(names):
  ret = set()
  for k in names:
    ret.add(getImgUrl(k))
  #return "".join(list(ret)) # TODO why not?
  return ret

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def slugify(value):
    # TODO not the right choice here.
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    import re
    value = unicodedata.normalize('NFKD', unicode(value)).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    return value

class Cacher(object):
  def __init__(self, directory, serializer = cPickle.dump, deserializer = cPickle.load, cacheSize = 50):
    self.directory = os.path.dirname(directory)
    if not os.path.isdir(self.directory):
      raise OSError("directory '%s' doesn't exist" % self.directory)
    self.serializer = serializer
    self.deserializer = deserializer
    self.cacheSize = cacheSize
    self.cache = {}
  def _addToCache(self, key, obj):
    if len(self.cache) >= self.cacheSize:
      # Evict one at random. #TODO #Lazy
      self.cache.pop(self.cache.keys()[random.randint(0, 9)])
    self.cache[key] = obj
  def getFilename(self, func, *args, **kwargs):
    return os.path.join(self.directory, slugify("%s_args=%s_kwargs=%s" % (func.__name__, str(args), str(sorted(kwargs.iteritems())))))
  def runMyKey(self, key, func, *args, **kwargs):
    if len(key) > 512:
      print "key too long!"
      exit(1)
    if key not in self.cache:
      # load from disk, or compute
      if os.path.isfile(key):
        obj = self.deserializer(open(key))
      else:
        print "Can't find key file %s" % str(key)
        obj = func(*args, **kwargs)
        self.serializer(obj, open(key, "w+"))
        if os.path.isfile(key):
          print "made file %s" % str(key)
        else:
          print "failed to make file %s" % str(key)
      self._addToCache(key, obj)
    return self.load(key)
  def runMyFile(self, filename, func, *args, **kwargs):
    return self.runMyKey(os.path.join(self.directory, filename), func, *args, **kwargs)
  def run(self, func, *args, **kwargs):
    key = self.getFilename(func, *args, **kwargs)
    return self.runMyKey(key, func, *args, **kwargs)
  def store(key, obj):
    if len(key) > 512:
      print "key too long!"
      exit(1)
    self.serializer(obj, open(key, "w+"))
    self._addToCache(key, obj)
  def load(self, key):
    if key not in self.cache and os.path.isfile(key):
      self._addToCache(key, self.deserializer(open(key)))
    return self.cache[key]
