import operator as op
import itertools

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
  return '<img src="https://s3.amazonaws.com/my89-frame-annotation/public/images_256/%s">' % (name)

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
