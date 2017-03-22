#!/bin/sh
echo "Epoch,D_Loss,G_Loss,D(x),D(G(z))_1,D(G(z))_2" > $1.csv
cat $1 | ./utils/filter_ganlog.py >> $1.csv
./utils/make_graphs.py $1.csv
