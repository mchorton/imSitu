#!/usr/bin/python
import re
import sys

_matchstring = "^\[(.*)/[0-9]*\]\[(.*)\] Loss_D: (.*) Loss_G: (.*) D\(x\): (.*) D\(G\(z\)\): (.*) / (.*)"
ganregex = re.compile(_matchstring)

for line in sys.stdin:
  line = line.rstrip("\n")
  match = ganregex.match(line)
  if match:
    print "%s,%s,%s,%s,%s,%s" % (
        str(float(match.group(1)) + eval("1. * " + match.group(2))), 
        float(match.group(3)),
        float(match.group(4)),
        float(match.group(5)),
        float(match.group(6)),
        float(match.group(7)))
  else:
    print "FAIL TO MATCH LINE %s" % line
