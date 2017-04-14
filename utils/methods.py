import os
def makeDirIfNeeded(filename):
  """
  Create the given directory, or the directory in which the file would be
  located.
  # filename: the name of the file or directory
  """
  directory = os.path.dirname(filename)
  if not os.path.exists(directory):
    os.makedirs(directory)

