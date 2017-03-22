#!/bin/sh
# First arg should be a directory where the log files are kept.
echo "<html>" > $1/allgraphs.html
find $1 -name "*.log" -exec echo {} \;
