import utils.mylogger as logging
def imgref(img):
  return '<img src="%s">' % (img)

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
  def addRow(self, arg):
    self.rows.append(arg)
  def save(self, outname):
    with open(outname, "w") as f:
      f.write(str(self))
  def __str__(self):
    output = ""
    output += self.header
    output += "<table>"
    for row in self.rows:
      output += "<tr>\n"
      for n, elem in enumerate(row):
        output += "<th>"
        output += "%s" % str(elem)
        output += "</th>\n"
      output += "</tr>\n"
    output += "</table>"
    output += self.footer
    return output

def pairToEqStr(pair):
    logging.getLogger(__name__).info(pair)
    logging.getLogger(__name__).info(map(str, pair))
    return "%s=%s" % tuple(map(str, pair))

class Element(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __str__(self):
        ret = "<"
        ret += " ".join(self.args)
        ret += " "
        ret += " ".join(map(pairToEqStr, self.kwargs.iteritems()))
        return ret + ">" # Should it be />? TODO

class ImgRef(Element):
    def __init__(self, **kwargs):
        """
        User should supply 'src=blah'
        But we won't sanity check.
        """
        super(ImgRef, self).__init__("img", **kwargs)
