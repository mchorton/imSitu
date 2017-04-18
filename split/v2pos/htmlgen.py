def imgref(img):
  return '<img src="%s">' % (img)

class HtmlMaker(object):
    def __init__(self):
        self._title = ""
        self.elements = []
    def addElement(self, element):
        """
        Add an element to the body of the HTML
        """
        self.elements.append(element)
    def toStr(self):
        ret = self.getPreamble()
        for elem in self.elements:
            ret += str(elem)
        ret += self.getCoda()
        return ret
    def addTitle(self, text):
        self._title = text
    def getPreamble(self):
        return (
                "<!DOCTYPE html>\n"
                "<html>\n"
                "<head>\n"
                "<title>%s</title>"
                "<style>\n"
                "table, th, td {\n"
                "      border: 1px solid black;\n"
                "}\n"
                "td { white-space:pre }\n"
                "</style>\n"
                "</head>\n"
                "<body>\n") % self._title
    def getCoda(self):
        return (
                "</body>\n"
                "</html>")
    def __str__(self):
        return self.toStr()
    def save(self, outname):
        with open(outname, "w") as f:
            f.write(str(self))

class HtmlTable(object):
    def __init__(self):
        self.rows = []
        self.style = ""
    def addRow(self, *args):
        self.rows.append(args)
    def toStr(self):
        ret = "<table>\n"
        for row in self.rows:
            ret += "  <tr>\n"
            if len(row) > 1:
                ret += "    <td>\n    "
            ret += "\n    </td>\n    <td>\n    ".join(map(str, row))
            if len(row) > 1:
                ret += "\n    </td>\n"
            ret += "  </tr>\n"
        return ret + "</table>\n"
    def __str__(self):
        return self.toStr()

def pairToEqStr(pair):
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
        kwargs["src"] = '"%s"' % kwargs["src"]
        super(ImgRef, self).__init__("img", **kwargs)

class HRef(object):
    def __init__(self, link, text):
        self._link = link
        self._text = text
    def __str__(self):
        return '<a href="%s">%s</a>' % (self._link, self._text)

class PhpTextFile(object):
    def __init__(self, filename):
        self.filename = filename
    def __str__(self):
        return '<div><p><?php echo file_get_contents("%s"); ?></p></div>\n' % self.filename

class Paragraph(object):
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return "<p>%s</p>\n" % str(self.text)

class Title(object):
    def __init__(self, text):
        self._text = text
    def __str__(self):
        return "<title>%s</title>" % self._text

class Heading(object):
    def __init__(self, level, text):
        self._level = level
        self._text = text
    def __str__(self):
        return "<h%d>%s</h%d>" % (self._level, self._text, self._level)
