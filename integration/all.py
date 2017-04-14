# This file defines integration tests.
# These are mostly scripts that I shouldn't break, e.g. this just makes sure
# that I don't e.g. delete needed functions, and is useful while developing.
# It doesn't make sure that the code functions as intended!
import os
import split.splitters as spsp
import split.rp_experiments as rpe
import pair_train.exp as exp
testdatadir = "testing/_int_spsptest"
distdir = "testing/_int_dist/"
vrndir = "testing/_int_vrndata/"
trainName = os.path.join(testdatadir, "zsTrain.json")
def main():
    spsp.splitTrainDevTestMinInTrain(testdatadir, test=True)

    rpe.generateAllDistVecStyle(
            distdir, trainName)

    vrnStub()
    exp.runTestExp()

def vrnStub():
    rpe.getVrnData(distdir, trainName, vrndir, thresh=float('inf'), freqthresh=1, blacklistprs = [], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True)

if __name__ == '__main__':
    main()
