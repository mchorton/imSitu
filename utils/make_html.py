import split.splitutils as sut
import os

def makeHTML(directoryName):
  """
  Creates HTML in the given directory, suitable for showing plots from that directory.
  """
  table = sut.HtmlTable()
  names = []
  # Iterate over all file names in the directory, sorted.
  for filename in os.listdir(directoryName):
    # TODO I could easily create the .csv and stuff in a function like this, but oh well.
    if filename.endswith(".jpg"):
      names.append(filename)

  names = sorted(names)
  for name in names:
    imgref = "<img src='%s'>" % name
    table.addRowNice(name, imgref)

  with open(os.path.join(directoryName, "all_graphs.html"), "w+") as out:
    out.write(str(table))

if __name__ == '__main__':
  import sys
  makeHTML(sys.argv[1])
