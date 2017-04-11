#!/bin/sh
# This file contains a call that will turn logs (in data/runs/ directory) into
# graphs, for gan-style training. It also makes an html file for viewing them.
# To also serve the html, you may want to use the utils/myserver class.

# $1 is the directory
DIR=$1
find $DIR -name "*.log" -exec ./utils/log2graph.sh {} \;
python utils/make_html.py $DIR
